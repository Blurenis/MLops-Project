# tests/unit/test_inference.py
"""
Unit tests for src/inference.py.

Covers
------
- Loading a saved model directory for inference
- Predicting labels and probabilities on short texts

Notes
-----
- Builds a temporary model dir from the tiny random BERT, writes label_map.json,
  then loads via the inference helpers.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np

# Make "src" importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: E402
from src.inference import load_model_for_inference, predict  # noqa: E402


TINY = "hf-internal-testing/tiny-random-bert"


def _make_temp_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tiny_model"
    d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TINY, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        TINY, num_labels=3, ignore_mismatched_sizes=True
    )
    tok.save_pretrained(str(d))
    mdl.save_pretrained(str(d))
    # Write an explicit label map
    label_map = {
        "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
        "label2id": {"negative": 0, "neutral": 1, "positive": 2},
    }
    (d / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return d


def test_inference_end_to_end(tmp_path: Path) -> None:
    model_dir = _make_temp_model_dir(tmp_path)
    model, tok, id2label = load_model_for_inference(str(model_dir))
    texts = ["awesome app", "crashes often"]
    out = predict(texts, model, tok, id2label, max_length=16)

    assert isinstance(out, list) and len(out) == 2
    for item in out:
        assert set(item.keys()) == {"label_id", "label_str", "probs"}
        probs = item["probs"]
        assert probs.shape[-1] == 3
        # each row softmax sums to 1
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
        assert item["label_str"] in {"negative", "neutral", "positive"}
