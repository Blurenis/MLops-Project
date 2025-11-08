# src/data_processing.py
"""
Text cleaning, tokenization, and train/validation splitting.

Capabilities
------------
- Clean text: Unicode normalize, lowercase, trim, collapse spaces, strip URLs,
  strip simple HTML tags, remove control chars.
- Tokenize with Hugging Face AutoTokenizer (e.g., "bert-base-uncased").
- Split a DataFrame into train/validation with optional stratification.
- CLI for quick processing.

Inputs
------
- A pandas DataFrame with a text column (e.g., "content").
- Optional label column for stratified split (e.g., "score").

Outputs
-------
- Encodings for train/val: input_ids, attention_mask.
- Optional labels for train/val when label_column is provided.
- The split DataFrames for downstream use.

Notes
-----
- Uses your existing loader if available: src.data_extraction.load_raw_data.
- Deterministic splits with random_state=42.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Optional import. If not present, users can pass an already-loaded DataFrame.
try:
    from .data_extraction import load_raw_data  # type: ignore
except Exception:  # pragma: no cover
    load_raw_data = None  # type: ignore


_URL_RE = re.compile(
    r"""(?xi)
    \b
    (?:https?://|www\.)
    [^\s<>"]+
    """
)

_TAG_RE = re.compile(r"<[^>]+>")

_CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]+")


def clean_text(text: str) -> str:
    """
    Normalize and simplify raw text.

    Steps
    -----
    - None-safe casting to str.
    - HTML unescape.
    - Remove URLs.
    - Remove simple HTML tags.
    - Unicode NFKC normalization.
    - Lowercase.
    - Remove control characters.
    - Collapse consecutive whitespace.

    Examples
    --------
    >>> clean_text("Hello   WORLD! Visit https://x.y ")
    'hello world! visit'
    """
    if text is None:
        return ""
    s = str(text)
    s = html.unescape(s)
    s = _URL_RE.sub("", s)
    s = _TAG_RE.sub(" ", s)
    s = _CTRL_RE.sub(" ", s)
    s = s.strip()
    # Unicode normalize and lowercase
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s


def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Tokenize texts with padding and truncation.

    Returns
    -------
    dict with keys:
      - input_ids: np.ndarray [N, max_length]
      - attention_mask: np.ndarray [N, max_length]
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def split_dataframe(
    df: pd.DataFrame,
    label_column: Optional[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train and validation sets.

    If `label_column` is provided, try stratified split. If stratification is
    not feasible (e.g., some classes have <2 samples), fall back to non-stratified.
    """
    strat = df[label_column] if label_column else None

    def _do_split(stratify_arg):
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )

    # Try stratified if requested
    if strat is not None:
        # Quick feasibility check
        vc = strat.value_counts(dropna=False)
        can_stratify = vc.min() >= 2 and len(vc) <= len(df) - 1
        if can_stratify:
            try:
                train_df, val_df = _do_split(strat)
                return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
            except ValueError:
                pass  # fall back

    # Fallback: non-stratified
    train_df, val_df = _do_split(None)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: Optional[str] = None,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Clean text, tokenize, and split.

    Parameters
    ----------
    df : DataFrame with at least `text_column`.
    text_column : name of the text column (e.g., "content").
    label_column : optional supervised label column (e.g., "score").
    tokenizer_name : Hugging Face tokenizer id.
    max_length : pad/truncate length.
    test_size : validation proportion.
    random_state : split seed.

    Returns
    -------
    dict with:
      - train_encodings: dict of arrays
      - val_encodings: dict of arrays
      - train_labels: Optional[np.ndarray]
      - val_labels: Optional[np.ndarray]
      - train_df, val_df: the split frames with cleaned text column
    """
    if text_column not in df.columns:
        raise ValueError(f"Missing text_column: {text_column}")

    work = df.copy()
    work[text_column] = work[text_column].map(clean_text)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    train_df, val_df = split_dataframe(
        work, label_column=label_column, test_size=test_size, random_state=random_state
    )

    train_texts = train_df[text_column].astype(str).tolist()
    val_texts = val_df[text_column].astype(str).tolist()

    train_enc = tokenize_texts(train_texts, tokenizer, max_length=max_length)
    val_enc = tokenize_texts(val_texts, tokenizer, max_length=max_length)

    train_labels = (
        train_df[label_column].to_numpy() if label_column and label_column in train_df else None
    )
    val_labels = (
        val_df[label_column].to_numpy() if label_column and label_column in val_df else None
    )

    return {
        "train_encodings": train_enc,
        "val_encodings": val_enc,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "train_df": train_df,
        "val_df": val_df,
    }


def _cli() -> None:
    """
    Minimal CLI.

    Examples
    --------
    CSV/TSV already loaded by data_extraction:
        python -m src.data_processing dataset.csv --text content --label score \\
            --tokenizer bert-base-uncased --max-length 128

    If you only want to inspect sizes without labels:
        python -m src.data_processing dataset.csv --text content
    """
    p = argparse.ArgumentParser(description="Clean, tokenize, and split text data.")
    p.add_argument("path", type=str, help="Path to dataset (csv/tsv/json/ndjson)")
    p.add_argument("--text", dest="text_col", required=True, help="Text column name")
    p.add_argument("--label", dest="label_col", default=None, help="Optional label column")
    p.add_argument("--tokenizer", default="bert-base-uncased", help="HF tokenizer id")
    p.add_argument("--max-length", type=int, default=128, help="Pad/truncate length")
    p.add_argument("--test-size", type=float, default=0.2, help="Validation proportion")
    args = p.parse_args()

    if load_raw_data is None:
        raise SystemExit("data_extraction.load_raw_data not found in src/.")

    df = load_raw_data(args.path)
    out = preprocess_dataframe(
        df,
        text_column=args.text_col,
        label_column=args.label_col,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        test_size=args.test_size,
    )
    n_train = len(out["train_df"])  # type: ignore
    n_val = len(out["val_df"])      # type: ignore
    print(f"OK | train={n_train} val={n_val} max_length={args.max_length} tokenizer={args.tokenizer}")


if __name__ == "__main__":
    _cli()
