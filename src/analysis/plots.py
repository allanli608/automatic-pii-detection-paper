"""Plotting utilities for training runs and ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def plot_metric_curves(
    run_name: str,
    metric_key: str = "eval_fbeta",
    output_dir: str = "outputs/analysis",
) -> Path:
    """Plot metric curves per epoch for each fold."""
    run_dir = Path("outputs") / run_name
    fold_paths = sorted(run_dir.glob("fold_*/trainer_state.json"))

    plt.figure(figsize=(8, 5))
    for path in fold_paths:
        data = json.loads(path.read_text())
        history = data.get("log_history", [])
        epochs = []
        values = []
        for entry in history:
            if metric_key in entry:
                epochs.append(entry.get("epoch", len(epochs) + 1))
                values.append(entry[metric_key])
        if values:
            plt.plot(epochs, values, label=path.parent.name)

    plt.title(f"{run_name}: {metric_key}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_key)
    plt.legend(loc="best")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_{metric_key}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_variant_comparison(
    run_names: Iterable[str],
    metric: str = "fbeta",
    output_dir: str = "outputs/analysis",
) -> Path:
    """Plot bar chart comparing variants by metric mean/std."""
    run_names = list(run_names)
    means: List[float] = []
    stds: List[float] = []

    for run_name in run_names:
        summary_path = Path("outputs") / run_name / "summary.json"
        data = json.loads(summary_path.read_text())
        stats = data.get("summary", {}).get(metric, {})
        means.append(stats.get("mean", 0.0))
        stds.append(stats.get("std", 0.0))

    plt.figure(figsize=(8, 4))
    plt.bar(run_names, means, yerr=stds, capsize=4)
    plt.ylabel(metric)
    plt.title(f"Variant comparison: {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_variants_{metric}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _load_id2label(run_name: str) -> dict:
    model_dir = Path("models") / run_name
    for info_path in model_dir.glob("fold_*/model_info.json"):
        data = json.loads(info_path.read_text())
        return {int(k): v for k, v in data["id2label"].items()}
    raise FileNotFoundError(f"No model_info.json found under {model_dir}")


def plot_confidence_histogram(
    run_name: str,
    output_dir: str = "outputs/analysis",
) -> Path:
    """Plot histogram of token-level confidence for PII vs O."""
    preds_dir = Path("outputs") / run_name / "preds"
    id2label = _load_id2label(run_name)

    pii_confidences: List[float] = []
    o_confidences: List[float] = []

    for path in preds_dir.glob("fold_*.npz"):
        data = np.load(path)
        probs = _softmax(data["logits"])
        labels = data["labels"]
        attention = data["attention_mask"]
        conf = np.max(probs, axis=-1)

        mask = (labels != -100) & (attention == 1)
        label_values = labels[mask]
        conf_values = conf[mask]

        for label, confidence in zip(label_values, conf_values):
            if id2label[int(label)] == "O":
                o_confidences.append(float(confidence))
            else:
                pii_confidences.append(float(confidence))

    plt.figure(figsize=(7, 4))
    plt.hist(o_confidences, bins=50, alpha=0.6, label="O")
    plt.hist(pii_confidences, bins=50, alpha=0.6, label="PII")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence histogram: {run_name}")
    plt.legend()
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_confidence_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
