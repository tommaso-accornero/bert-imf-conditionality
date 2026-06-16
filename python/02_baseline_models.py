# python/02_baseline_models.py
# Phase 3: TF-IDF baseline models — performance floor for BERT comparison.
# Run from project root: python python/02_baseline_models.py
#
# Inputs:  data/processed/train.csv, test.csv, label_encoder_13cat.json
# Outputs: results/tables/baseline_results.csv

import json
import random

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────

train = pd.read_csv(PROCESSED_DIR / "train.csv")
test  = pd.read_csv(PROCESSED_DIR / "test.csv")

with open(PROCESSED_DIR / "label_encoder_13cat.json") as f:
    encoder = json.load(f)
idx2label = encoder['idx2label']

X_train, y_train = train['text'].values, train['label_13cat'].values
X_test,  y_test  = test['text'].values,  test['label_13cat'].values

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Labels: {sorted(set(y_train))}\n")

# ── 2. TF-IDF vectoriser (shared by both models) ───────────────────────────────

print("Fitting TF-IDF vectoriser (max_features=10,000)...")
tfidf = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # log-scaling reduces impact of very high-frequency terms
    min_df=2,             # ignore terms that appear in only one document
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}\n")

# ── 3. Logistic Regression ─────────────────────────────────────────────────────

print("Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver='lbfgs',
    random_state=SEED,
    n_jobs=-1,
)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)

lr_acc     = accuracy_score(y_test, lr_preds)
lr_macro_f1 = f1_score(y_test, lr_preds, average='macro')

print(f"Logistic Regression — Accuracy: {lr_acc:.4f} | Macro-F1: {lr_macro_f1:.4f}")

# ── 4. Random Forest ───────────────────────────────────────────────────────────

print("\nTraining Random Forest (this takes a few minutes)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)

rf_acc      = accuracy_score(y_test, rf_preds)
rf_macro_f1 = f1_score(y_test, rf_preds, average='macro')

print(f"Random Forest      — Accuracy: {rf_acc:.4f} | Macro-F1: {rf_macro_f1:.4f}")

# ── 5. Per-category F1 table ───────────────────────────────────────────────────

label_names = [idx2label[str(i)] for i in sorted(set(y_test))]

lr_report = classification_report(y_test, lr_preds, target_names=label_names, output_dict=True)
rf_report = classification_report(y_test, rf_preds, target_names=label_names, output_dict=True)

per_cat = pd.DataFrame({
    'category':   label_names,
    'n_test':     [sum(y_test == i) for i in sorted(set(y_test))],
    'lr_f1':      [round(lr_report[c]['f1-score'], 3) for c in label_names],
    'rf_f1':      [round(rf_report[c]['f1-score'], 3) for c in label_names],
}).sort_values('n_test', ascending=False)

print("\nPer-category F1 (test set):")
print(per_cat.to_string(index=False))

# ── 6. Summary table ───────────────────────────────────────────────────────────

summary = pd.DataFrame([
    {'model': 'TF-IDF + Logistic Regression', 'accuracy': round(lr_acc, 4), 'macro_f1': round(lr_macro_f1, 4)},
    {'model': 'TF-IDF + Random Forest',       'accuracy': round(rf_acc, 4), 'macro_f1': round(rf_macro_f1, 4)},
])

print("\nSummary:")
print(summary.to_string(index=False))
print(f"\nBERT target: macro-F1 >= 0.80")
print(f"Baseline floor (best model): macro-F1 = {max(lr_macro_f1, rf_macro_f1):.4f}")

# ── 7. Save ────────────────────────────────────────────────────────────────────

per_cat['lr_f1_vs_bert_target'] = per_cat['lr_f1'].apply(lambda x: 'below 0.55' if x < 0.55 else 'ok')
per_cat.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
summary.to_csv(RESULTS_DIR / "baseline_summary.csv", index=False)

print(f"\nSaved:")
print(f"  results/tables/baseline_results.csv")
print(f"  results/tables/baseline_summary.csv")
print("\nPhase 3 baseline models complete.")
