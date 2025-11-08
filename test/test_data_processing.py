# tests/unit/test_data_processing.py
"""
Unit tests for src/data_processing.py

Scope
-----
- Text cleaning: URL removal, lowercasing, whitespace collapse.
- Tokenization parity with Hugging Face reference calls.
- Stratified split size and label distribution preservation.
- End-to-end preprocess on a tiny DataFrame.

Layout assumptions
------------------
- src/data_processing.py
- src/data_extraction.py
- tests/unit/test_data_processing.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_processing import (  # noqa: E402
    clean_text,
    preprocess_dataframe,
    tokenize_texts,
)
from transformers import AutoTokenizer  # noqa: E402


def test_clean_text_basic() -> None:
    s = "Hello   WORLD! Visit https://ex.am/ple <b>Now</b>\n"
    out = clean_text(s)
    # URL removed, tags stripped, lowercased, spaces collapsed
    assert "http" not in out
    assert "<b>" not in out
    assert out == "hello world! visit now"


def test_tokenize_parity_with_hf() -> None:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    texts = ["hello world", "tokenization check"]
    # Our pipeline
    ours = tokenize_texts(texts, tok, max_length=12)
    # Reference encoding
    ref = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=12,
        return_attention_mask=True,
        return_tensors="np",
    )
    assert np.array_equal(ours["input_ids"], ref["input_ids"])
    assert np.array_equal(ours["attention_mask"], ref["attention_mask"])
    # CLS and SEP positions sanity
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    assert ours["input_ids"][0, 0] == cls_id
    assert sep_id in ours["input_ids"][0]


def test_stratified_split_distribution() -> None:
    # Create imbalanced labels to verify stratification
    data = {
        "content": [f"sample {i}" for i in range(50)],
        "score": [0] * 40 + [1] * 10,
    }
    df = pd.DataFrame(data)
    out = preprocess_dataframe(
        df,
        text_column="content",
        label_column="score",
        tokenizer_name="bert-base-uncased",
        max_length=8,
        test_size=0.2,
    )
    train_df = out["train_df"]
    val_df = out["val_df"]
    # Check sizes
    assert len(train_df) == 40
    assert len(val_df) == 10
    # Check distribution approximately equal via proportion
    train_prop = train_df["score"].mean()
    val_prop = val_df["score"].mean()
    assert abs(train_prop - val_prop) < 0.05


def test_end_to_end_shapes() -> None:
    df = pd.DataFrame(
        {
            "content": ["A GREAT App!!", "bad... experience", "okay-ish"],
            "score": [5, 1, 3],
        }
    )
    out = preprocess_dataframe(
        df,
        text_column="content",
        label_column="score",
        tokenizer_name="bert-base-uncased",
        max_length=16,
        test_size=1 / 3,  # 2 train, 1 val
    )
    train_enc = out["train_encodings"]
    val_enc = out["val_encodings"]
    # Shapes
    assert train_enc["input_ids"].shape[0] == 2
    assert train_enc["input_ids"].shape[1] == 16
    assert val_enc["input_ids"].shape == (1, 16)
    # Labels returned
    assert out["train_labels"] is not None
    assert out["val_labels"] is not None
