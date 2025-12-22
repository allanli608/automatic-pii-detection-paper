"""K-fold training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.train_fold import train_one_fold, StatusHook


def _aggregate_metrics(metrics_by_fold: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"folds": metrics_by_fold}

    metrics_keys = set()
    for metrics in metrics_by_fold.values():
        metrics_keys.update(metrics.keys())

    stats: Dict[str, Dict[str, float]] = {}
    for key in metrics_keys:
        values = [metrics.get(key) for metrics in metrics_by_fold.values() if key in metrics]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((val - mean) ** 2 for val in values) / len(values)
        stats[key] = {
            "mean": float(mean),
            "std": float(var ** 0.5),
        }

    summary["summary"] = stats
    return summary


def train_kfold(config, variant, status_hook: Optional[StatusHook] = None) -> Dict[str, Any]:
    """Train all folds for a variant and return summary metrics."""
    metrics_by_fold: Dict[int, Dict[str, Any]] = {}

    for fold_index in range(config.NUM_FOLDS):
        metrics_by_fold[fold_index] = train_one_fold(
            fold_index=fold_index,
            config=config,
            variant=variant,
            status_hook=status_hook,
        )

    summary = _aggregate_metrics(metrics_by_fold)
    summary_path = Path(config.OUTPUT_DIR_BASE) / config.RUN_NAME / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
