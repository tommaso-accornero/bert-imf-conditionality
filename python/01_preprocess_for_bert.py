# python/01_preprocess_for_bert.py
# Phase 3: Deduplicate, encode labels, stratified train/val/test split.
# Run from project root: python python/01_preprocess_for_bert.py
#
# Inputs:  data/processed/imf_monitor_clean.csv
# Outputs: data/processed/train.csv, val.csv, test.csv
#          data/processed/label_encoder_13cat.json, label_encoder_7cat.json

import json
import random

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print(f"Random seed: {SEED}  (fix this value to reproduce the exact split)")

PROCESSED_DIR = Path("data/processed")

CATEGORIES_13 = ['FIN', 'DEB', 'FP', 'RTP', 'EXT', 'SOE', 'LAB', 'PRI', 'INS', 'POV', 'SP', 'OTH', 'ENV']

COLLAPSE_MAP = {
    'FP':  'FISCAL',        'RTP': 'FISCAL',        'DEB': 'FISCAL',
    'FIN': 'FINANCIAL',
    'EXT': 'EXTERNAL',
    'SOE': 'PRIVATISATION', 'PRI': 'PRIVATISATION',
    'LAB': 'LABOUR',
    'SP':  'SOCIAL',        'POV': 'SOCIAL',
    'INS': 'GOVERNANCE',    'ENV': 'GOVERNANCE',     'OTH': 'GOVERNANCE',
}

CATEGORIES_7 = sorted(set(COLLAPSE_MAP.values()))

# ── 1. Load ────────────────────────────────────────────────────────────────────

df = pd.read_csv(PROCESSED_DIR / "imf_monitor_clean.csv", low_memory=False)
print(f"Loaded: {len(df):,} rows")

# ── 2. Deduplicate on (country, year, text) ────────────────────────────────────
#
# R/01 retained 1,089 cross-program duplicates: same condition text, same country,
# same year, different arrangement_id. Removed here before the split so the same
# condition instance cannot appear in both train and test.
#
# We do NOT deduplicate on text alone. A pre-run check found only 13 texts (0.07%)
# that carry conflicting labels across programs — all genuine annotation ambiguity.
# Boilerplate reuse across countries/years is consistent in labeling and represents
# real pattern BERT should learn. Text-only dedup would remove 16,314 rows and cut
# DEB training examples by 91% for no material quality gain.

n_before = len(df)
df = df.drop_duplicates(subset=['country', 'year', 'text'], keep='first').reset_index(drop=True)
print(f"Deduplication: {n_before - len(df):,} cross-program duplicates dropped ({n_before:,} -> {len(df):,})")

# ── 3. Validate labels ─────────────────────────────────────────────────────────

unknown = set(df['category'].unique()) - set(CATEGORIES_13)
if unknown:
    raise ValueError(f"Unknown categories in data: {unknown}")

print(f"\n13-category distribution:")
print(df['category'].value_counts().to_string())

# ── 4. Add 7-category labels ───────────────────────────────────────────────────
#
# imf_monitor_clean.csv does not contain collapsed_7cat (R/01 saves only raw columns).
# If a future version of R/01 adds the column, the assert below will catch any
# disagreement between the R derivation and the Python COLLAPSE_MAP.

if 'collapsed_7cat' in df.columns:
    derived = df['category'].map(COLLAPSE_MAP)
    mismatch = (df['collapsed_7cat'] != derived).sum()
    if mismatch > 0:
        raise ValueError(
            f"collapsed_7cat already in CSV but disagrees with COLLAPSE_MAP on {mismatch} rows. "
            "Check that R/02 and Python use identical collapse logic."
        )
    print("collapsed_7cat already in CSV and matches COLLAPSE_MAP — keeping existing column.")
else:
    df['collapsed_7cat'] = df['category'].map(COLLAPSE_MAP)
    print("collapsed_7cat derived from COLLAPSE_MAP.")

assert df['collapsed_7cat'].isna().sum() == 0, "Some 13-cat labels not covered by COLLAPSE_MAP"

# ── 5. Build and save label encoders ──────────────────────────────────────────

label2idx_13 = {cat: i for i, cat in enumerate(CATEGORIES_13)}
label2idx_7  = {cat: i for i, cat in enumerate(CATEGORIES_7)}

df['label_13cat'] = df['category'].map(label2idx_13)
df['label_7cat']  = df['collapsed_7cat'].map(label2idx_7)

encoder_13 = {
    'label2idx': label2idx_13,
    'idx2label': {str(v): k for k, v in label2idx_13.items()}
}
encoder_7 = {
    'label2idx': label2idx_7,
    'idx2label': {str(v): k for k, v in label2idx_7.items()}
}

with open(PROCESSED_DIR / "label_encoder_13cat.json", 'w') as f:
    json.dump(encoder_13, f, indent=2)
with open(PROCESSED_DIR / "label_encoder_7cat.json", 'w') as f:
    json.dump(encoder_7, f, indent=2)

print(f"\n13-cat label mapping: {label2idx_13}")
print(f"7-cat label mapping:  {label2idx_7}")

# ── 6. Stratified train / val / test split (70 / 15 / 15) ─────────────────────
#
# Stratified on 13-cat labels so minority classes (ENV: ~199, SP: ~370, POV: ~518)
# are proportionally represented in every split.
#
# Limitation (document in Chapter 3): deduplication removes identical texts across
# arrangements (hard leakage). It does not remove non-identical conditions from the
# same arrangement, which can be split between train and test. Conditions from the
# same program share country context and stylistic register; this within-arrangement
# clustering may modestly inflate performance estimates relative to a true
# arrangement-level split. An arrangement-level split would severely undersample
# minority categories and is not viable given the class sizes.

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df['label_13cat'],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df['label_13cat'],
    random_state=SEED
)

total = len(df)
print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):,}  ({len(train_df)/total*100:.1f}%)")
print(f"  Val:   {len(val_df):,}   ({len(val_df)/total*100:.1f}%)")
print(f"  Test:  {len(test_df):,}   ({len(test_df)/total*100:.1f}%)")

# ── 7. Verify proportions are consistent across splits ────────────────────────

print("\nClass proportions (13-cat) — full / train / val / test:")
dist = pd.DataFrame({
    'full':  df['category'].value_counts(normalize=True).round(3),
    'train': train_df['category'].value_counts(normalize=True).round(3),
    'val':   val_df['category'].value_counts(normalize=True).round(3),
    'test':  test_df['category'].value_counts(normalize=True).round(3),
}).loc[CATEGORIES_13]
print(dist.to_string())

# ── 8. Save splits ─────────────────────────────────────────────────────────────

OUTPUT_COLS = ['text', 'label_13cat', 'label_7cat', 'category', 'collapsed_7cat', 'year', 'country']

train_df[OUTPUT_COLS].to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df[OUTPUT_COLS].to_csv(PROCESSED_DIR  / "val.csv",   index=False)
test_df[OUTPUT_COLS].to_csv(PROCESSED_DIR / "test.csv",  index=False)

print(f"\nSaved to {PROCESSED_DIR}/:")
print(f"  train.csv              ({len(train_df):,} rows)")
print(f"  val.csv                ({len(val_df):,} rows)")
print(f"  test.csv               ({len(test_df):,} rows)")
print(f"  label_encoder_13cat.json")
print(f"  label_encoder_7cat.json")
print("\nPhase 3 preprocessing complete.")
