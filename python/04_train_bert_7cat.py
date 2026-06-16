# python/04_train_bert_7cat.py
# Phase 3: Fine-tune bert-base-uncased for 7-category classification (robustness check).
# Run from project root: python python/04_train_bert_7cat.py
#
# Inputs:  data/processed/train.csv, val.csv, test.csv
#          data/processed/label_encoder_7cat.json
# Outputs: models/bert_7cat/              (saved model + tokenizer)
#          results/tables/bert_7cat_per_category_f1.csv

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS     = 4
LR         = 3e-5
NUM_LABELS = 7

PROCESSED_DIR  = Path("data/processed")
MODELS_DIR     = Path("models/bert_7cat")
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR    = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── 2. Load data ───────────────────────────────────────────────────────────────

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")

with open(PROCESSED_DIR / "label_encoder_7cat.json") as f:
    encoder = json.load(f)
idx2label   = encoder['idx2label']
label_names = [idx2label[str(i)] for i in range(NUM_LABELS)]

print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Labels: {label_names}")

# ── 3. Tokenise ────────────────────────────────────────────────────────────────

print(f"\nLoading tokeniser: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
    )

def to_hf_dataset(df):
    ds = Dataset.from_pandas(
        df[['text', 'label_7cat']].rename(columns={'label_7cat': 'labels'})
    )
    ds = ds.map(tokenize, batched=True, desc="Tokenising")
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return ds

print("Tokenising splits...")
train_dataset = to_hf_dataset(train_df)
val_dataset   = to_hf_dataset(val_df)
test_dataset  = to_hf_dataset(test_df)

# ── 4. Class weights ───────────────────────────────────────────────────────────
#
# FISCAL = 47.2% of training data — still worth weighting even with 7 classes.

raw_weights  = compute_class_weight(
    'balanced',
    classes=np.arange(NUM_LABELS),
    y=train_df['label_7cat'].values
)
class_weights = torch.tensor(raw_weights, dtype=torch.float32)
print(f"\nClass weights (balanced):")
for name, w in zip(label_names, raw_weights):
    print(f"  {name:>15}  {w:.3f}")

# ── 5. Model ───────────────────────────────────────────────────────────────────

print(f"\nLoading model: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# ── 6. Custom Trainer with weighted cross-entropy loss ─────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ── 7. Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'macro_f1': round(f1_score(labels, preds, average='macro'), 4),
        'accuracy': round(accuracy_score(labels, preds), 4),
    }

# ── 8. Training arguments ──────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
    seed=SEED,
    fp16=False,
    bf16=False,
)

# ── 9. Train ───────────────────────────────────────────────────────────────────

print(f"\nTraining BERT 7-cat ({EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE}, max_len={MAX_LENGTH})")
print("Val macro-F1 will print at end of each epoch.\n")

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ── 10. Evaluate on test set ───────────────────────────────────────────────────

print("\nEvaluating on held-out test set...")
test_output = trainer.predict(test_dataset)

logits = test_output.predictions
y_test = test_output.label_ids
preds  = np.argmax(logits, axis=-1)

test_macro_f1 = f1_score(y_test, preds, average='macro')
test_accuracy = accuracy_score(y_test, preds)

print(f"\nTest results:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Macro-F1:  {test_macro_f1:.4f}  (13-cat model scored 0.8976 — expect higher here)")

report = classification_report(y_test, preds, target_names=label_names, output_dict=True)
per_cat = pd.DataFrame([
    {
        'category':  name,
        'n_test':    int(sum(y_test == i)),
        'precision': round(report[name]['precision'], 3),
        'recall':    round(report[name]['recall'], 3),
        'f1':        round(report[name]['f1-score'], 3),
    }
    for i, name in enumerate(label_names)
]).sort_values('n_test', ascending=False)

print("\nPer-category F1 (test set):")
print(per_cat.to_string(index=False))

probs      = F.softmax(torch.tensor(logits), dim=-1).numpy()
confidence = probs.max(axis=-1)
print(f"\nPrediction confidence on test set:")
print(f"  Mean:    {confidence.mean():.3f}")
print(f"  Median:  {np.median(confidence):.3f}")
print(f"  < 0.70:  {(confidence < 0.70).sum()} predictions ({(confidence < 0.70).mean()*100:.1f}%)")

# ── 11. Save model ─────────────────────────────────────────────────────────────

print(f"\nSaving model to {MODELS_DIR}/")
trainer.save_model(str(MODELS_DIR))
tokenizer.save_pretrained(str(MODELS_DIR))

# ── 12. Save results ───────────────────────────────────────────────────────────

per_cat.to_csv(RESULTS_DIR / "bert_7cat_per_category_f1.csv", index=False)

summary = pd.DataFrame([{
    'model':      'BERT 7-cat',
    'accuracy':   round(test_accuracy, 4),
    'macro_f1':   round(test_macro_f1, 4),
    'epochs':     EPOCHS,
    'lr':         LR,
    'max_length': MAX_LENGTH,
    'batch_size': BATCH_SIZE,
    'seed':       SEED,
}])
summary.to_csv(RESULTS_DIR / "bert_7cat_summary.csv", index=False)

print(f"\nSaved:")
print(f"  results/tables/bert_7cat_per_category_f1.csv")
print(f"  results/tables/bert_7cat_summary.csv")
print(f"  models/bert_7cat/")
print("\nPhase 3 BERT 7-cat training complete.")
