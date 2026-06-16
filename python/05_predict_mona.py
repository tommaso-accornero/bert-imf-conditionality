# python/05_predict_mona.py
# Phase 4: Run trained BERT 13-cat model on MONA 2020-2024 conditions.
# Run from project root: python python/05_predict_mona.py
#
# Inputs:  data/processed/mona_with_labels.csv  (includes lookup labels from R/02)
#          data/processed/label_encoder_13cat.json
#          models/bert_13cat/                    (fine-tuned model)
# Outputs: data/final/mona_classified.csv

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
FINAL_DIR     = Path("data/final")

MAX_LENGTH   = 128
INFER_BATCH  = 64
CONF_THRESHOLD = 0.70   # predictions below this are flagged for manual review
FINAL_DIR.mkdir(parents=True, exist_ok=True)

COLLAPSE_MAP = {
    'FP':  'FISCAL',        'RTP': 'FISCAL',        'DEB': 'FISCAL',
    'FIN': 'FINANCIAL',
    'EXT': 'EXTERNAL',
    'SOE': 'PRIVATISATION', 'PRI': 'PRIVATISATION',
    'LAB': 'LABOUR',
    'SP':  'SOCIAL',        'POV': 'SOCIAL',
    'INS': 'GOVERNANCE',    'ENV': 'GOVERNANCE',     'OTH': 'GOVERNANCE',
}

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

# ── 3. Load MONA 2020-2024 ─────────────────────────────────────────────────────
#
# Load mona_with_labels.csv (not mona_clean.csv) so the lookup labels from R/02
# travel alongside BERT predictions. Python/06 uses both to measure agreement.

mona = pd.read_csv(PROCESSED_DIR / "mona_with_labels.csv", low_memory=False)

# Extract test_year from 'Test Date' — more accurate than approval year for time series.
# 'Program Prior Action' and other non-dates parse to NaT; fall back to approval year.
mona['test_year'] = pd.to_datetime(mona['Test Date'], errors='coerce').dt.year
mona['test_year'] = mona['test_year'].fillna(mona['year']).astype(int)

# Filter to approval years 2020-2024, then restrict to test_year <= 2024
# (excludes conditions agreed now but due in the future)
mona_target = mona[
    mona['year'].between(2020, 2024) &
    (mona['test_year'] >= 2020) &
    (mona['test_year'] <= 2024)
].copy().reset_index(drop=True)

print(f"Rows after filtering to test_year <= 2024: {len(mona_target):,} (removed {len(mona[mona['year'].between(2020,2024)]) - len(mona_target):,} future-dated conditions)")
print(f"Test year distribution:\n{mona_target['test_year'].value_counts().sort_index().to_string()}")

print(f"\nMONA 2020-2024: {len(mona_target):,} conditions")
print(f"Countries: {mona_target['country'].nunique()}")
print(f"Year distribution:\n{mona_target['year'].value_counts().sort_index().to_string()}")

texts = mona_target['text'].fillna('').values

# ── 4. Load model and tokeniser ────────────────────────────────────────────────

print(f"\nLoading fine-tuned model from {MODELS_DIR}/")
tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))
model     = BertForSequenceClassification.from_pretrained(str(MODELS_DIR))
model.eval()
model.to(device)
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 5. Tokenise ────────────────────────────────────────────────────────────────

print(f"\nTokenising {len(texts):,} conditions (max_length={MAX_LENGTH})...")
encodings = tokenizer(
    texts.tolist(),
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='pt',
)
print("Tokenisation complete.")

# ── 6. Batch inference ─────────────────────────────────────────────────────────

print(f"\nRunning inference (batch_size={INFER_BATCH})...")
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

# ── 7. Decode predictions ──────────────────────────────────────────────────────

bert_category    = [idx2label[str(i)] for i in pred_idx]
bert_collapsed   = [COLLAPSE_MAP[c] for c in bert_category]
needs_review     = confidence < CONF_THRESHOLD

print(f"\nInference complete.")
print(f"Confidence distribution:")
print(f"  Mean:    {confidence.mean():.3f}")
print(f"  Median:  {np.median(confidence):.3f}")
print(f"  < {CONF_THRESHOLD}: {needs_review.sum():,} predictions ({needs_review.mean()*100:.1f}%) flagged for review")

# ── 8. Assemble output ─────────────────────────────────────────────────────────

output = mona_target[[
    'arrangement_number', 'country', 'year', 'test_year', 'arrangement_type',
    'text', 'descriptor', 'key_code',
    'imf_category', 'collapsed_7cat',          # lookup labels from R/02
]].copy()

output['bert_category']   = bert_category
output['bert_7cat']       = bert_collapsed
output['bert_confidence'] = confidence.round(4)
output['needs_review']    = needs_review

# ── 9. Summary ─────────────────────────────────────────────────────────────────

print("\nBERT 13-cat distribution (MONA 2020-2024):")
print(
    output['bert_category']
    .value_counts()
    .rename_axis('category')
    .reset_index(name='n')
    .assign(pct=lambda d: (d['n'] / d['n'].sum() * 100).round(1))
    .to_string(index=False)
)

print("\nBERT 7-cat distribution (MONA 2020-2024):")
print(
    output['bert_7cat']
    .value_counts()
    .rename_axis('category')
    .reset_index(name='n')
    .assign(pct=lambda d: (d['n'] / d['n'].sum() * 100).round(1))
    .to_string(index=False)
)

print("\nLookup vs BERT agreement (7-cat):")
agree = (output['collapsed_7cat'] == output['bert_7cat']).sum()
total = len(output)
print(f"  Agreement: {agree:,} / {total:,} ({agree/total*100:.1f}%)")

print("\nDisagreement breakdown (top 10 lookup → BERT pairs):")
disagreements = output[output['collapsed_7cat'] != output['bert_7cat']]
if len(disagreements) > 0:
    pairs = (
        disagreements
        .groupby(['collapsed_7cat', 'bert_7cat'])
        .size()
        .reset_index(name='n')
        .sort_values('n', ascending=False)
        .head(10)
    )
    print(pairs.to_string(index=False))

# ── 10. Save ───────────────────────────────────────────────────────────────────

output.to_csv(FINAL_DIR / "mona_classified.csv", index=False)
print(f"\nSaved: data/final/mona_classified.csv ({len(output):,} rows)")
print("\nPhase 4 MONA inference complete.")
