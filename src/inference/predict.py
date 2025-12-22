"""Prediction helpers for fold models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from src.models.multi_dropout import MultiDropoutTokenClassifier


def load_model_for_fold(
    run_name: str,
    fold: int,
    device: Optional[torch.device] = None,
    model_dir_base: str = "models",
) -> Tuple[torch.nn.Module, object]:
    """Load a saved model and tokenizer for a fold."""
    model_dir = Path(model_dir_base) / run_name / f"fold_{fold}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    wrapper_meta = model_dir / "wrapper_meta.json"
    if wrapper_meta.exists():
        model = MultiDropoutTokenClassifier.from_pretrained(str(model_dir))
    else:
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    return model, tokenizer


def _pad_batch(tensor: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
    if tensor.size(1) == target_len:
        return tensor
    pad_len = target_len - tensor.size(1)
    if tensor.dim() == 3:
        return F.pad(tensor, (0, 0, 0, pad_len), value=pad_value)
    return F.pad(tensor, (0, pad_len), value=pad_value)


def predict_logits(
    dataset,
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict logits for a dataset and return logits, labels, attention mask."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    logits_list = []
    labels_list = []
    attention_list = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            labels = batch["labels"].detach().cpu()
            attention_mask = batch["attention_mask"].detach().cpu()

            logits_list.append(logits)
            labels_list.append(labels)
            attention_list.append(attention_mask)

    max_len = max(t.size(1) for t in logits_list)
    logits = torch.cat([_pad_batch(t, max_len, 0.0) for t in logits_list], dim=0)
    labels = torch.cat([_pad_batch(t, max_len, -100) for t in labels_list], dim=0)
    attention_mask = torch.cat([_pad_batch(t, max_len, 0) for t in attention_list], dim=0)

    return logits.numpy(), labels.numpy(), attention_mask.numpy()


def save_fold_logits(
    run_name: str,
    fold: int,
    logits: np.ndarray,
    labels: np.ndarray,
    attention_mask: np.ndarray,
    output_dir_base: str = "outputs",
) -> Path:
    """Save fold logits and labels to NPZ."""
    out_dir = Path(output_dir_base) / run_name / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold_{fold}.npz"
    np.savez_compressed(out_path, logits=logits, labels=labels, attention_mask=attention_mask)
    return out_path
