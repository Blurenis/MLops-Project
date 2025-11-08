"""
BERT fine-tuning for sentiment classification (Trainer-based).

Overview
--------
- Clean and map labels from raw review scores (or use an explicit column).
- Tokenize text with a Hugging Face tokenizer.
- Create a sequence classification model (BERT).
- Train with the Trainer API and evaluate on a validation set.
- Save artifacts to an output directory for inference:
  - model weights, tokenizer, config, and label_map.json.

Defaults
--------
- Model: "bert-base-uncased" (change to "hf-internal-testing/tiny-random-bert"
  when running quick tests without a GPU).
- Label mapping strategy: by default, convert 1..5 star scores to three classes:
    <=2 -> negative, 3 -> neutral, >=4 -> positive.

Inputs expected
---------------
A DataFrame with:
- `text_column` e.g. "content"
- Either:
  * numeric review `score` (1..5) to be mapped to 3 classes, or
  * a categorical/encoded `label_column` that already contains class ids or names.

CLI
---
Example with score-based mapping:
    python -m src.model --path dataset.csv --text content --score score \
        --output ./artifacts/bert-sentiment --epochs 1 --batch-size 16

Example with an explicit label column (string or int):
    python -m src.model --path dataset.csv --text content --label label \
        --output ./artifacts/bert-sentiment --epochs 1 --batch-size 16

Notes
-----
- Uses deterministic split with random_state=42.
- Works on CPU. If CUDA is available, mixed precision (fp16) is enabled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Optional import from your earlier loader. If absent, we fallback to pandas.read_csv.
try:
    from .data_extraction import load_raw_data  # type: ignore
except Exception:
    load_raw_data = None  # type: ignore


# ----------------------------
# Label handling
# ----------------------------

def map_scores_to_3way_label(score: Union[int, float]) -> str:
    """
    Map a star score (1..5) to a 3-class sentiment label.
    <=2 -> 'negative', 3 -> 'neutral', >=4 -> 'positive'
    """
    if pd.isna(score):
        return "neutral"
    try:
        s = float(score)
    except Exception:
        return "neutral"
    if s <= 2:
        return "negative"
    if s >= 4:
        return "positive"
    return "neutral"


def build_label_encoder(
    series: pd.Series, explicit_order: Optional[List[str]] = None
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build {label_str -> id} and {id -> label_str} mappings.

    If `explicit_order` is given, use that order. Otherwise infer from sorted unique labels.
    """
    if explicit_order:
        labels = list(explicit_order)
    else:
        labels = sorted({str(x) for x in series.astype(str).tolist()})
    str2id = {lab: i for i, lab in enumerate(labels)}
    id2str = {i: lab for lab, i in str2id.items()}
    return str2id, id2str


# ----------------------------
# Dataset
# ----------------------------

@dataclass
class EncodedItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None


class HFDataset(Dataset):
    """
    Simple dataset wrapping tokenized inputs and optional labels.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int,
        label_ids: Optional[List[int]] = None,
    ):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = (
            torch.tensor(label_ids, dtype=torch.long) if label_ids is not None else None
        )

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ----------------------------
# Core pipeline
# ----------------------------

def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV/TSV/JSON dataset. Uses your data_extraction if available."""
    if load_raw_data is None:
        # Fallback: pandas infers separators reasonably well for CSV/TSV
        return pd.read_csv(path)
    return load_raw_data(path)


def prepare_data(
    df: pd.DataFrame,
    text_column: str,
    label_column: Optional[str],
    score_column: Optional[str],
    test_size: float,
    max_length: int,
    tokenizer_name: str,
    label_order: Optional[List[str]] = None,
) -> Tuple[HFDataset, HFDataset, Dict[str, int], Dict[int, str], AutoTokenizer]:
    """
    Clean, label-encode, tokenize, and split.
    Returns train_ds, val_ds, str2id, id2str, tokenizer.
    """
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    # Build text series
    texts = df[text_column].astype(str).fillna("").tolist()

    # Build label strings
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Missing label column: {label_column}")
        label_strs = df[label_column].astype(str).tolist()
    elif score_column:
        if score_column not in df.columns:
            raise ValueError(f"Missing score column: {score_column}")
        label_strs = [map_scores_to_3way_label(x) for x in df[score_column]]
    else:
        raise ValueError("Provide either --label or --score to define labels.")

    # Encode labels
    str2id, id2str = build_label_encoder(pd.Series(label_strs), explicit_order=label_order)
    label_ids = [str2id[s] for s in label_strs]

    # Split
    train_texts, val_texts, train_y, val_y = train_test_split(
        texts, label_ids, test_size=test_size, random_state=42, stratify=label_ids
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Datasets
    train_ds = HFDataset(train_texts, tokenizer, max_length=max_length, label_ids=train_y)
    val_ds = HFDataset(val_texts, tokenizer, max_length=max_length, label_ids=val_y)

    return train_ds, val_ds, str2id, id2str, tokenizer


def create_model(
    model_name: str,
    num_labels: int,
):
    """
    Create a sequence classification model. `ignore_mismatched_sizes=True` allows
    swapping tiny checkpoints that shipped with a different head size.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


def train(
    train_ds: HFDataset,
    val_ds: HFDataset,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_labels: int,
    output_dir: Union[str, Path],
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
) -> None:
    """
    Train and save to `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(model_name, num_labels)

    # CPU works; if CUDA is available, Trainer will use it.
    fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=50,
        report_to=[],          # pas de trackers
        # fp16 facultatif; commente-le si ta version ne le supporte pas
        # fp16=torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def save_label_map(id2str: Dict[int, str], output_dir: Union[str, Path]) -> None:
    """
    Save label id <-> string mapping for inference.
    """
    payload = {
        "id2label": {str(k): v for k, v in id2str.items()},
        "label2id": {v: int(k) for k, v in id2str.items()},
    }
    Path(output_dir, "label_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="Fine-tune BERT for sentiment classification.")
    p.add_argument("--path", required=True, help="Path to dataset file")
    p.add_argument("--text", dest="text_col", required=True, help="Text column name (e.g., content)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", dest="label_col", help="Explicit label column (string or int)")
    group.add_argument("--score", dest="score_col", help="Numeric score column to map to 3 classes")
    p.add_argument("--model", default="bert-base-uncased", help="HF model id")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id (defaults to model id)")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--output", required=True, help="Directory to save model artifacts")
    args = p.parse_args()

    # Load
    df = load_dataframe(args.path)

    # Prepare
    train_ds, val_ds, str2id, id2str, tok = prepare_data(
        df=df,
        text_column=args.text_col,
        label_column=args.label_col,
        score_column=args.score_col,
        test_size=args.test_size,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer or args.model,
    )

    # Train
    train(
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tok,
        model_name=args.model,
        num_labels=len(str2id),
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save label map
    save_label_map(id2str, args.output)

    print(f"OK | saved model and tokenizer to: {args.output}")


if __name__ == "__main__":
    _cli()
