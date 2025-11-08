"""
Inference utilities for the fine-tuned sentiment model.

What this provides
------------------
- `load_model_for_inference(model_dir)` to load a saved directory from training.
- `predict(texts, model, tokenizer, max_length)` to return label and probabilities.

The directory must contain:
- model files (pytorch_model.bin, config.json)
- tokenizer files (vocab.txt or equivalent, tokenizer.json, tokenizer_config.json)
- label_map.json created during training (id2label + label2id)

CLI
---
Example:
    python -m src.inference --model-dir ./artifacts/bert-sentiment \\
        --texts "great app" "awful experience"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def load_model_for_inference(model_dir: str) -> Tuple[torch.nn.Module, AutoTokenizer, dict]:
    """
    Load a fine-tuned model directory plus tokenizer and label mapping.
    """
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    label_map_path = Path(model_dir) / "label_map.json"
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in label_map.get("id2label", {}).items()}
    else:
        # fallback: use config if available
        id2label = getattr(model.config, "id2label", None) or {}
        id2label = {int(k): v for k, v in id2label.items()}

    return model, tokenizer, id2label


def predict(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    id2label: dict,
    max_length: int = 128,
) -> List[dict]:
    """
    Predict labels and probabilities for a list of texts.

    Returns
    -------
    List[dict] with keys:
      - label_id: int
      - label_str: str
      - probs: np.ndarray [num_labels]
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    device = next(model.parameters()).device
    for k in ("input_ids", "attention_mask"):
        enc[k] = enc[k].to(device)

    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()

    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)

    out = []
    for i, p in enumerate(preds):
        label_id = int(p)
        label_str = id2label.get(label_id, str(label_id))
        out.append(
            {"label_id": label_id, "label_str": label_str, "probs": probs[i]}
        )
    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run sentiment inference on new texts.")
    p.add_argument("--model-dir", required=True, help="Path to saved model directory")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--texts", nargs="+", required=True, help="Texts to score")
    args = p.parse_args()

    model, tok, id2label = load_model_for_inference(args.model_dir)
    results = predict(args.texts, model, tok, id2label, max_length=args.max_length)
    for t, r in zip(args.texts, results):
        print(f"{t} -> {r['label_str']} | probs={np.round(r['probs'], 4)}")


if __name__ == "__main__":
    _cli()
