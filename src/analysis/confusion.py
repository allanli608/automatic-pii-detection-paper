"""Confusion matrix utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def _load_id2label(run_name: str) -> Dict[int, str]:
    model_dir = Path("models") / run_name
    for info_path in model_dir.glob("fold_*/model_info.json"):
        data = json.loads(info_path.read_text())
        return {int(k): v for k, v in data["id2label"].items()}
    raise FileNotFoundError(f"No model_info.json found under {model_dir}")


def build_confusion_matrix(run_name: str) -> Dict[str, object]:
    """Aggregate token-level confusion matrix across folds."""
    preds_dir = Path("outputs") / run_name / "preds"
    id2label = _load_id2label(run_name)
    num_labels = len(id2label)

    confusion = np.zeros((num_labels, num_labels), dtype=int)

    for path in preds_dir.glob("fold_*.npz"):
        data = np.load(path)
        logits = data["logits"]
        labels = data["labels"]
        attention = data["attention_mask"]

        preds = np.argmax(logits, axis=-1)
        mask = (labels != -100) & (attention == 1)

        for true_id, pred_id in zip(labels[mask], preds[mask]):
            confusion[int(true_id), int(pred_id)] += 1

    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}

    for idx in range(num_labels):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision[id2label[idx]] = float(tp / (tp + fp + 1e-12))
        recall[id2label[idx]] = float(tp / (tp + fn + 1e-12))

    return {
        "labels": [id2label[i] for i in range(num_labels)],
        "confusion": confusion.tolist(),
        "precision": precision,
        "recall": recall,
    }
