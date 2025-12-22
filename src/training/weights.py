"""Class weight builders for token classification."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable

import torch


def make_uniform_weights(label2id: Dict[str, int]) -> torch.Tensor:
    """Return uniform weights for each label."""
    return torch.ones(len(label2id), dtype=torch.float)


def make_o_weighting(label2id: Dict[str, int], o_weight: float) -> torch.Tensor:
    """Return weights with the O label down-weighted."""
    weights = torch.ones(len(label2id), dtype=torch.float)
    if "O" in label2id:
        weights[label2id["O"]] = float(o_weight)
    return weights


def make_dynamic_weights_from_datasets(
    label2id: Dict[str, int],
    datasets: Iterable,
    ignore_index: int = -100,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """Compute balanced class weights from token frequency counts."""
    label_counts: Counter[int] = Counter()
    total_valid_tokens = 0

    for dataset in datasets:
        for idx in range(len(dataset)):
            item = dataset[idx]
            labels = item["labels"] if isinstance(item, dict) else item.labels
            valid = [int(label) for label in labels if int(label) != ignore_index]
            label_counts.update(valid)
            total_valid_tokens += len(valid)

    num_classes = len(label2id)
    weights = torch.ones(num_classes, dtype=torch.float)

    for label_name, label_id in label2id.items():
        count = label_counts.get(label_id, 0)
        if count > 0:
            weights[label_id] = total_valid_tokens / (num_classes * count)
        else:
            weights[label_id] = 1.0

    weights = torch.clamp(weights, min=clamp_min, max=clamp_max)
    return weights
