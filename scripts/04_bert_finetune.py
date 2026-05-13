"""
04_bert_finetune.py
Phase 5 — Fine-tune BERT for token classification (NER).

Trains on your manually annotated sentences from Label Studio or CSV.
Uses your RTX 4050 GPU for fast training (~20 mins).

BEFORE RUNNING:
  Complete annotation in Label Studio or annotation_sheet.csv first.
  Then run: python scripts/04_bert_finetune.py

If annotation not yet done, run with --use-baseline flag to train
on baseline predictions (weaker but demonstrates the pipeline):
  python scripts/04_bert_finetune.py --use-baseline

Usage:
    python scripts/04_bert_finetune.py
    python scripts/04_bert_finetune.py --use-baseline
    python scripts/04_bert_finetune.py --epochs 5 --lr 2e-5
"""

import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

ROOT        = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "data" / "annotations"
MODELS      = ROOT / "models"
RESULTS     = ROOT / "results"
MODELS.mkdir(parents=True, exist_ok=True)

# ── Label scheme ──────────────────────────────────────────────────────────────
LABELS = [
    "O",
    "B-SPECIES",  "I-SPECIES",
    "B-BEHAVIOR", "I-BEHAVIOR",
    "B-HABITAT",  "I-HABITAT",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotated_label_studio(path: Path) -> list[dict]:
    """Load completed Label Studio export JSON."""
    with open(path) as f:
        tasks = json.load(f)

    samples = []
    for task in tasks:
        text     = task['data']['text']
        tokens   = text.split()
        char_labels = ['O'] * len(text)

        annotations = task.get('annotations', [])
        if not annotations:
            continue

        for ann in annotations[0].get('result', []):
            if ann.get('type') != 'labels':
                continue
            val   = ann['value']
            start = val['start']
            end   = val['end']
            label = val['labels'][0]
            if start < len(char_labels):
                char_labels[start] = f'B-{label}'
                for ci in range(start + 1, min(end, len(char_labels))):
                    char_labels[ci] = f'I-{label}'

        token_labels = _align_tokens_to_labels(text, tokens, char_labels)
        samples.append({"tokens": tokens, "labels": token_labels,
                        "text": text})
    return samples


def load_annotated_csv(path: Path) -> list[dict]:
    """
    Load completed annotation CSV.
    Converts semicolon-separated entity strings back to token labels.
    """
    import pandas as pd
    df = pd.read_csv(path)
    samples = []

    for _, row in df.iterrows():
        text   = str(row['text'])
        tokens = text.split()

        char_labels = ['O'] * len(text)

        for col, label in [('SPECIES','SPECIES'),
                            ('BEHAVIOR','BEHAVIOR'),
                            ('HABITAT','HABITAT')]:
            cell = str(row.get(col, ''))
            if cell and cell != 'nan':
                for entity in cell.split(';'):
                    entity = entity.strip()
                    if not entity:
                        continue
                    start = text.lower().find(entity.lower())
                    if start != -1:
                        end = start + len(entity)
                        char_labels[start] = f'B-{label}'
                        for ci in range(start + 1, min(end, len(char_labels))):
                            char_labels[ci] = f'I-{label}'

        token_labels = _align_tokens_to_labels(text, tokens, char_labels)
        samples.append({"tokens": tokens, "labels": token_labels,
                        "text": text})
    return samples


def load_baseline_as_training(path: Path, n: int = 300) -> list[dict]:
    """
    Use baseline NER predictions as silver training data.
    Weaker than human annotation but demonstrates full pipeline.
    """
    with open(path) as f:
        predictions = json.load(f)

    random.seed(42)
    selected = random.sample(predictions, min(n, len(predictions)))
    samples  = []

    for pred in selected:
        text   = pred['text']
        tokens = text.split()
        char_labels = ['O'] * len(text)

        for ent in pred.get('entities', []):
            start = ent['start_char']
            end   = ent['end_char']
            label = ent['label']
            if start < len(char_labels):
                char_labels[start] = f'B-{label}'
                for ci in range(start + 1, min(end, len(char_labels))):
                    char_labels[ci] = f'I-{label}'

        token_labels = _align_tokens_to_labels(text, tokens, char_labels)
        samples.append({"tokens": tokens, "labels": token_labels,
                        "text": text})
    return samples


def _align_tokens_to_labels(text: str, tokens: list[str],
                              char_labels: list[str]) -> list[str]:
    """Map character-level labels to token-level labels."""
    token_labels = []
    pos = 0
    for token in tokens:
        start = text.find(token, pos)
        if start == -1:
            token_labels.append('O')
            pos += len(token) + 1
            continue
        label = char_labels[start] if start < len(char_labels) else 'O'
        token_labels.append(label)
        pos = start + len(token)
    return token_labels


# ── BERT NER Model ────────────────────────────────────────────────────────────

def train_bert_ner(samples: list[dict],
                   model_name: str = "bert-base-cased",
                   epochs: int = 5,
                   lr: float = 2e-5,
                   batch_size: int = 16,
                   val_split: float = 0.15):
    """
    Fine-tune BERT for token classification.
    Saves model to models/bert_ner/
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (BertTokenizerFast, BertForTokenClassification,
                                  get_linear_schedule_with_warmup)
        from torch.optim import AdamW
    except ImportError:
        print("ERROR: transformers or torch not installed.")
        print("Run: pip install transformers torch")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # ── Dataset ───────────────────────────────────────────────────────────────
    class NERDataset(Dataset):
        def __init__(self, samples, tokenizer, max_len=128):
            self.samples   = samples
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            tokens = sample['tokens']
            labels = sample['labels']

            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt',
            )

            # Align labels to subword tokens
            word_ids    = encoding.word_ids()
            label_ids   = []
            prev_word   = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)   # special token
                elif word_id != prev_word:
                    label_ids.append(LABEL2ID.get(labels[word_id]
                                     if word_id < len(labels) else 'O', 0))
                else:
                    # Subword continuation: use I- label
                    lbl = labels[word_id] if word_id < len(labels) else 'O'
                    if lbl.startswith('B-'):
                        lbl = 'I-' + lbl[2:]
                    label_ids.append(LABEL2ID.get(lbl, 0))
                prev_word = word_id

            return {
                'input_ids':      encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'token_type_ids': encoding['token_type_ids'].squeeze(),
                'labels':         torch.tensor(label_ids, dtype=torch.long),
            }

    # Split train/val
    random.shuffle(samples)
    n_val    = max(1, int(len(samples) * val_split))
    tr_data  = samples[n_val:]
    val_data = samples[:n_val]
    print(f"  Train samples : {len(tr_data)}")
    print(f"  Val samples   : {len(val_data)}")

    tr_dataset  = NERDataset(tr_data,  tokenizer)
    val_dataset = NERDataset(val_data, tokenizer)
    tr_loader   = DataLoader(tr_dataset,  batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size)

    # Load model
    print(f"\n  Loading model: {model_name}")
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"\n  Training for {epochs} epochs...")
    print(f"  {'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10}")
    print(f"  {'-'*40}")

    best_f1 = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

                preds  = outputs.logits.argmax(-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                for pred_seq, label_seq in zip(preds, labels):
                    for p, l in zip(pred_seq, label_seq):
                        if l != -100:
                            all_preds.append(ID2LABEL[p])
                            all_labels.append(ID2LABEL[l])

        val_loss /= len(val_loader)
        val_f1    = compute_f1(all_labels, all_preds)

        history['train_loss'].append(round(tr_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_f1'].append(round(val_f1, 4))

        print(f"  {epoch+1:<6} {tr_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODELS / "bert_ner")
            tokenizer.save_pretrained(MODELS / "bert_ner")

    print(f"\n  Best val F1: {best_f1:.4f}")
    print(f"  Model saved: {MODELS / 'bert_ner'}")

    # Save history
    with open(RESULTS / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return model, tokenizer, history


def compute_f1(true_labels: list[str], pred_labels: list[str]) -> float:
    """Span-level F1 for NER (entity-level, not token-level)."""
    def get_spans(labels):
        spans, start, current = set(), None, None
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                if current:
                    spans.add((start, i - 1, current))
                start, current = i, label[2:]
            elif label.startswith('I-') and current:
                pass
            else:
                if current:
                    spans.add((start, i - 1, current))
                start, current = None, None
        if current:
            spans.add((start, len(labels) - 1, current))
        return spans

    true_spans = get_spans(true_labels)
    pred_spans = get_spans(pred_labels)

    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    return f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-baseline', action='store_true',
                        help='Train on baseline predictions (before annotation)')
    parser.add_argument('--epochs',  type=int,   default=5)
    parser.add_argument('--lr',      type=float, default=2e-5)
    parser.add_argument('--batch',   type=int,   default=16)
    parser.add_argument('--model',   type=str,   default='bert-base-cased')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 — BERT Fine-tuning for NER")
    print("=" * 60)

    # Load training data
    if args.use_baseline:
        print("\n  Mode: Training on baseline predictions (silver labels)")
        samples = load_baseline_as_training(
            ROOT / "results" / "baseline_predictions.json", n=300)
    else:
        # Try Label Studio export first, then CSV
        ls_annotated = ANNOTATIONS / "annotated.json"
        csv_annotated = ANNOTATIONS / "annotation_sheet.csv"

        if ls_annotated.exists():
            print("\n  Mode: Training on Label Studio annotations (gold labels)")
            samples = load_annotated_label_studio(ls_annotated)
        elif csv_annotated.exists():
            print("\n  Mode: Training on CSV annotations (gold labels)")
            samples = load_annotated_csv(csv_annotated)
        else:
            print("\n  No annotation file found.")
            print("  Run with --use-baseline to train on baseline predictions.")
            print("  Or complete annotation first:")
            print("    Label Studio: data/annotations/label_studio_tasks.json")
            print("    CSV:          data/annotations/annotation_sheet.csv")
            return

    print(f"  Training samples: {len(samples)}")

    # Label distribution
    all_labels = [l for s in samples for l in s['labels']]
    label_dist = Counter(all_labels)
    print("\n  Label distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label:<15} {count:>6}")

    # Train
    result = train_bert_ner(
        samples,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )

    if result[0] is not None:
        model, tokenizer, history = result
        print("\n  Training history:")
        for i, (tl, vl, vf) in enumerate(zip(
                history['train_loss'],
                history['val_loss'],
                history['val_f1'])):
            print(f"    Epoch {i+1}: loss={tl:.4f} val_loss={vl:.4f} val_f1={vf:.4f}")
        print("\n✓ BERT fine-tuning complete.")
    else:
        print("\n  Install dependencies and retry:")
        print("  pip install transformers torch")


if __name__ == "__main__":
    main()
