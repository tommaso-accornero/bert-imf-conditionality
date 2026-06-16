# python/06_evaluate_temporal.py
# Phase 4: Temporal validation — manual coding of 100 MONA 2020-2024 conditions.
# Run from project root: python python/06_evaluate_temporal.py
#
# This script has two modes depending on which files exist:
#
#   MODE 1 (first run): exports 100 conditions for manual coding.
#     Inputs:  data/final/mona_classified.csv
#     Outputs: results/validation/validation_sample.csv
#     Action:  open the CSV, fill in the 'manual_category' column, save as
#              results/validation/validation_sample_coded.csv
#
#   MODE 2 (after coding): compares manual labels to BERT predictions.
#     Inputs:  results/validation/validation_sample_coded.csv
#     Outputs: results/validation/temporal_validation_report.csv
#     Action:  reports BERT accuracy on the hand-coded sample (temporal drift test)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report

FINAL_DIR      = Path("data/final")
VALIDATION_DIR = Path("results/validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

SEED           = 42

CATEGORIES_13 = [
    'FIN', 'DEB', 'FP', 'RTP', 'EXT', 'SOE',
    'LAB', 'PRI', 'INS', 'POV', 'SP', 'OTH', 'ENV'
]

# ── MODE 2: compare if coded file already exists ───────────────────────────────

coded_path = VALIDATION_DIR / "validation_sample_coded.csv"

if coded_path.exists():
    print("Found validation_sample_coded.csv — running MODE 2 (comparison).\n")

    coded = pd.read_csv(coded_path)

    if 'manual_category' not in coded.columns:
        raise ValueError("'manual_category' column not found. Fill this column before running Mode 2.")

    missing = coded['manual_category'].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have no manual_category. They will be excluded.")
        coded = coded.dropna(subset=['manual_category'])

    # Handle dual-coded entries (e.g. "FP / INS") — extract primary (first) choice.
    # Coder indicated ~70% confidence in first choice, 30% in second.
    coded['is_uncertain'] = coded['manual_category'].str.contains('/', na=False)
    n_uncertain = coded['is_uncertain'].sum()
    if n_uncertain > 0:
        print(f"Dual-coded entries (uncertain): {n_uncertain} — using primary (first) choice.")
        coded['manual_category'] = coded['manual_category'].apply(
            lambda x: x.split('/')[0].strip() if '/' in str(x) else x
        )

    unknown = set(coded['manual_category'].unique()) - set(CATEGORIES_13)
    if unknown:
        raise ValueError(f"Unknown category values in manual_category: {unknown}\nValid values: {CATEGORIES_13}")

    y_manual = coded['manual_category'].values
    y_bert   = coded['bert_category'].values

    acc       = accuracy_score(y_manual, y_bert)
    macro_f1  = f1_score(y_manual, y_bert, average='macro', zero_division=0)

    print(f"Temporal validation results (n={len(coded)}):")
    print(f"  BERT accuracy vs manual:  {acc:.3f}")
    print(f"  BERT macro-F1 vs manual:  {macro_f1:.3f}")
    print()

    # Agreement by category
    report = classification_report(
        y_manual, y_bert,
        labels=CATEGORIES_13,
        zero_division=0,
        output_dict=True
    )
    per_cat = pd.DataFrame([
        {
            'category': cat,
            'n_manual': int(sum(y_manual == cat)),
            'agreement_f1': round(report[cat]['f1-score'], 3),
        }
        for cat in CATEGORIES_13
        if sum(y_manual == cat) > 0
    ]).sort_values('n_manual', ascending=False)

    print("Per-category agreement:")
    print(per_cat.to_string(index=False))

    # Disagreement table
    disagreements = coded[y_manual != y_bert][['text', 'manual_category', 'bert_category', 'bert_confidence']]
    print(f"\nDisagreements: {len(disagreements)} / {len(coded)} ({len(disagreements)/len(coded)*100:.1f}%)")
    if len(disagreements) > 0:
        print(disagreements[['manual_category', 'bert_category', 'bert_confidence']].to_string(index=False))

    # Save report
    per_cat.to_csv(VALIDATION_DIR / "temporal_validation_report.csv", index=False)
    print(f"\nSaved: results/validation/temporal_validation_report.csv")
    print("\nPhase 4 temporal validation complete.")

else:
    # ── MODE 1: export sample for manual coding ────────────────────────────────

    print("No coded file found — running MODE 1 (export sample for manual coding).\n")

    classified_path = FINAL_DIR / "mona_classified.csv"
    if not classified_path.exists():
        raise FileNotFoundError(
            "data/final/mona_classified.csv not found. Run python/05_predict_mona.py first."
        )

    df = pd.read_csv(classified_path)
    print(f"Loaded mona_classified.csv: {len(df):,} rows (2020-2024)")

    # Stratified sample: 20 conditions per year (2020-2024) = 100 total
    # Stratified to ensure temporal coverage across all 5 years.
    frames = []
    for yr in sorted(df['year'].unique()):
        yr_df = df[df['year'] == yr]
        frames.append(yr_df.sample(n=min(20, len(yr_df)), random_state=SEED))
    sample = pd.concat(frames).reset_index(drop=True)

    print(f"Sampled {len(sample)} conditions ({sample['year'].value_counts().sort_index().to_dict()})")

    # Add blank manual_category column for hand-coding
    sample['manual_category'] = ''

    # Keep only columns needed for coding + reference
    output_cols = [
        'country', 'year', 'descriptor', 'text',
        'bert_category', 'bert_confidence',
        'manual_category',                    # FILL THIS IN
    ]
    sample[output_cols].to_csv(VALIDATION_DIR / "validation_sample.csv", index=False)

    print(f"\nSaved: results/validation/validation_sample.csv")
    print()
    print("Next steps:")
    print("  1. Open results/validation/validation_sample.csv")
    print("  2. For each row, read the 'text' column and assign a category")
    print(f"  3. Valid categories: {CATEGORIES_13}")
    print("  4. Save as results/validation/validation_sample_coded.csv")
    print("  5. Re-run this script to compute accuracy")
