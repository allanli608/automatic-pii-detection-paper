"""Metric helpers for token classification."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    id2label: Dict[int, str],
) -> Dict[str, float]:
    """Compute strict F5 score for token classification."""
    predictions, labels = eval_pred

    preds = np.argmax(predictions, axis=-1)

    mask = labels != -100
    valid_preds = preds[mask]
    valid_labels = labels[mask]

    tp = 0
    fp = 0
    fn = 0

    for p_id, t_id in zip(valid_preds, valid_labels):
        p_str = id2label[p_id]
        t_str = id2label[t_id]

        if t_str != "O":
            if p_str == t_str:
                tp += 1
            else:
                fn += 1

        if p_str != "O":
            if p_str != t_str:
                fp += 1

    epsilon = 1e-12
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    beta = 5.0
    f5 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + epsilon)

    return {
        "fbeta": float(f5),
        "precision": float(precision),
        "recall": float(recall),
    }
