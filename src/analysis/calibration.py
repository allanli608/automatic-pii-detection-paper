"""Calibration utilities for token classification probabilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    attention_mask: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute expected calibration error for token predictions."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    mask = (labels != -100) & (attention_mask == 1)

    confidences = confidences[mask]
    predictions = predictions[mask]
    labels = labels[mask]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_acc = np.zeros(num_bins)
    bin_conf = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not np.any(in_bin):
            continue
        bin_acc[i] = np.mean(predictions[in_bin] == labels[in_bin])
        bin_conf[i] = np.mean(confidences[in_bin])
        ece += np.abs(bin_acc[i] - bin_conf[i]) * np.mean(in_bin)

    return float(ece), bin_acc, bin_conf
