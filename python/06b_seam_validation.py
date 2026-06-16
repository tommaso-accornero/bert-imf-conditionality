# python/06b_seam_validation.py
# Methodology seam validation: does BERT produce the same program-level scope
# as hand-coding on the same 2015-2019 IMF Monitor arrangements?
#
# Without this test, a reviewer can argue the 2019→2020 boundary is a method
# artefact. If BERT and hand-coding agree on scope for 2015-2019 programs,
# the method is consistent across the seam.
#
# Run from project root: python python/06b_seam_validation.py
#
# Inputs:  data/processed/imf_monitor_clean.csv
#          data/processed/label_encoder_13cat.json
#          models/bert_13cat/
# Outputs: results/validation/seam_validation.csv
#          results/validation/seam_validation_summary.txt

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

MODELS_DIR    = Path("models/bert_13cat")
PROCESSED_DIR = Path("data/processed")
VALIDATION_DIR = Path("results/validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH  = 128
INFER_BATCH = 64
SEAM_YEARS  = (2015, 2019)   # 5-year window immediately before the 2020 MONA period

# ── 1. Device ──────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# ── 2. Load label encoder ──────────────────────────────────────────────────────

with open(PROCESSED_DIR / "label_encoder_13cat.json") as f:
    encoder = json.load(f)
idx2label = encoder['idx2label']

# ── 3. Load IMF Monitor 2015-2019 ─────────────────────────────────────────────

imf = pd.read_csv(PROCESSED_DIR / "imf_monitor_clean.csv", low_memory=False)
seam_df = imf[
    imf['year'].between(SEAM_YEARS[0], SEAM_YEARS[1])
].copy().reset_index(drop=True)

print(f"\nIMF Monitor {SEAM_YEARS[0]}-{SEAM_YEARS[1]}: {len(seam_df):,} conditions")
print(f"Arrangements: {seam_df['arrangement_id'].nunique()}")
print(f"Countries: {seam_df['country'].nunique()}")

# ── 4. Compute hand-coded scope (ground truth) ────────────────────────────────

handcoded_scope = (
    seam_df
    .groupby(['country', 'arrangement_id', 'year'])
    .agg(
        n_conditions_hand  = ('category', 'count'),
        n_policy_areas_hand = ('category', 'nunique'),
    )
    .reset_index()
)

print(f"\nProgram-years (hand-coded): {len(handcoded_scope):,}")
print(f"Mean hand-coded scope: {handcoded_scope['n_policy_areas_hand'].mean():.2f}")

# ── 5. Load model ──────────────────────────────────────────────────────────────

print(f"\nLoading fine-tuned model from {MODELS_DIR}/")
tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))
model     = BertForSequenceClassification.from_pretrained(str(MODELS_DIR))
model.eval()
model.to(device)

# ── 6. Tokenise and run inference ─────────────────────────────────────────────

texts = seam_df['text'].fillna('').values

print(f"\nTokenising {len(texts):,} conditions...")
encodings = tokenizer(
    texts.tolist(),
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt',
)

print("Running inference...")
all_logits = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), INFER_BATCH), desc="Predicting"):
        input_ids      = encodings['input_ids'][i:i+INFER_BATCH].to(device)
        attention_mask = encodings['attention_mask'][i:i+INFER_BATCH].to(device)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())

logits     = torch.cat(all_logits, dim=0)
probs      = F.softmax(logits, dim=-1).numpy()
pred_idx   = probs.argmax(axis=-1)
confidence = probs.max(axis=-1)

seam_df['bert_category']   = [idx2label[str(i)] for i in pred_idx]
seam_df['bert_confidence'] = confidence

# ── 7. Compute BERT scope ─────────────────────────────────────────────────────

bert_scope = (
    seam_df
    .groupby(['country', 'arrangement_id', 'year'])
    .agg(
        n_policy_areas_bert = ('bert_category', 'nunique'),
        mean_confidence     = ('bert_confidence', 'mean'),
    )
    .reset_index()
)

# ── 8. Compare ────────────────────────────────────────────────────────────────

comparison = handcoded_scope.merge(bert_scope, on=['country', 'arrangement_id', 'year'])
comparison['scope_diff']    = comparison['n_policy_areas_bert'] - comparison['n_policy_areas_hand']
comparison['scope_abs_diff'] = comparison['scope_diff'].abs()
comparison['exact_match']   = comparison['scope_diff'] == 0
comparison['within_1']      = comparison['scope_abs_diff'] <= 1

n = len(comparison)
exact_pct  = comparison['exact_match'].mean() * 100
within1_pct = comparison['within_1'].mean() * 100
mae        = comparison['scope_abs_diff'].mean()
bias       = comparison['scope_diff'].mean()

print(f"\n=== Seam Validation Results ({SEAM_YEARS[0]}-{SEAM_YEARS[1]}) ===")
print(f"Program-years compared: {n:,}")
print(f"Exact match (BERT scope = hand-coded scope): {exact_pct:.1f}%")
print(f"Within 1 policy area:                        {within1_pct:.1f}%")
print(f"Mean absolute error (policy areas):          {mae:.3f}")
print(f"Bias (BERT - hand-coded, mean):              {bias:.3f}")
print(f"  Positive bias = BERT finds more categories than hand-coding")
print(f"  Negative bias = BERT finds fewer categories than hand-coding")

# Per-year breakdown
print(f"\nPer-year breakdown:")
year_summary = comparison.groupby('year').agg(
    n             = ('exact_match', 'count'),
    exact_pct     = ('exact_match', 'mean'),
    within1_pct   = ('within_1', 'mean'),
    mae           = ('scope_abs_diff', 'mean'),
    mean_bias     = ('scope_diff', 'mean'),
).round(3)
print(year_summary.to_string())

# ── 9. Interpretation ─────────────────────────────────────────────────────────

print(f"\n=== Interpretation ===")
if within1_pct >= 80:
    print(f"PASS: {within1_pct:.1f}% of program-years within 1 policy area.")
    print("The method seam is clean. BERT scope and hand-coded scope are consistent")
    print("for 2015-2019 programs. The 2019→2020 boundary is not a method artefact.")
elif within1_pct >= 65:
    print(f"MARGINAL: {within1_pct:.1f}% within 1 policy area.")
    print("Some disagreement exists. Inspect high-disagreement cases before finalising.")
    print("Add a note in Chapter 3 quantifying this validation.")
else:
    print(f"FAIL: only {within1_pct:.1f}% within 1 policy area.")
    print("Substantial disagreement between BERT and hand-coding on 2015-2019 programs.")
    print("Investigate before using BERT predictions for the mission creep time series.")

if abs(bias) > 0.3:
    direction = "overestimates" if bias > 0 else "underestimates"
    print(f"\nWARNING: BERT systematically {direction} scope by {abs(bias):.2f} policy areas.")
    print("This would create a spurious trend at the 2019→2020 boundary.")
    print("Consider a bias-correction or note this as a limitation in Chapter 3.")
else:
    print(f"\nBias: {bias:.3f} — no systematic over/under-estimation.")

# ── 10. Save ───────────────────────────────────────────────────────────────────

comparison.to_csv(VALIDATION_DIR / "seam_validation.csv", index=False)

summary_lines = [
    f"Seam Validation: IMF Monitor {SEAM_YEARS[0]}-{SEAM_YEARS[1]}",
    f"Program-years compared: {n}",
    f"Exact match: {exact_pct:.1f}%",
    f"Within 1 policy area: {within1_pct:.1f}%",
    f"MAE: {mae:.3f}",
    f"Bias (BERT - hand-coded): {bias:.3f}",
]
with open(VALIDATION_DIR / "seam_validation_summary.txt", 'w') as f:
    f.write("\n".join(summary_lines))

print(f"\nSaved:")
print(f"  results/validation/seam_validation.csv")
print(f"  results/validation/seam_validation_summary.txt")
print("\nSeam validation complete.")
