# tests/unit/test_mode.py
"""
Unit tests for src/model.py (Trainer-based pipeline).

Covers
------
- Score→label mapping
- Label encoder ordering
- Data preparation shapes and tokenizer load
- Model forward pass shape (no training)

Notes
-----
- Uses the tiny random BERT for speed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import pandas as pd

# Make "src" importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (  # noqa: E402
    map_scores_to_3way_label,
    build_label_encoder,
    prepare_data,
    create_model,
)


TINY = "hf-internal-testing/tiny-random-bert"


def test_score_mapping_3way() -> None:
    vals = [1, 2, 3, 4, 5, None, "x"]
    out = [map_scores_to_3way_label(v) for v in vals]
    # Expected: <=2 negative, 3 neutral, >=4 positive, unknown -> neutral
    assert out == ["negative", "negative", "neutral", "positive", "positive", "neutral", "neutral"]


def test_label_encoder_order() -> None:
    ser = pd.Series(["neutral", "negative", "positive", "negative"])
    str2id, id2str = build_label_encoder(ser, explicit_order=["negative", "neutral", "positive"])
    assert str2id == {"negative": 0, "neutral": 1, "positive": 2}
    assert id2str == {0: "negative", 1: "neutral", 2: "positive"}


def test_prepare_data_shapes_and_tokenizer() -> None:
    df = pd.DataFrame(
        {
            "content": [f"sample {i}" for i in range(20)],
            "score": [1] * 8 + [3] * 6 + [5] * 6,
        }
    )
    train_ds, val_ds, str2id, id2str, tok = prepare_data(
        df=df,
        text_column="content",
        label_column=None,
        score_column="score",
        test_size=0.2,
        max_length=16,
        tokenizer_name=TINY,
    )
    # Splits
    assert len(train_ds) == 16
    assert len(val_ds) == 4
    # Labels and tokenizer presence
    assert set(str2id.keys()) == {"negative", "neutral", "positive"}
    assert tok is not None


def test_model_forward_shape() -> None:
    # Minimal forward using tiny model
    model = create_model(TINY, num_labels=3)
    # Dummy tokenized batch (2 items, seq len 12)
    input_ids = torch.randint(low=0, high=100, size=(2, 12))
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    assert logits.shape == (2, 3)
