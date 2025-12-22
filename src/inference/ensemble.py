"""Ensembling helpers for token classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from src.metrics.metrics import compute_metrics


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _load_fold_logits(run_name: str, fold: int, output_dir_base: str) -> Dict[str, np.ndarray]:
    path = Path(output_dir_base) / run_name / "preds" / f"fold_{fold}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    data = np.load(path)
    return {"logits": data["logits"], "labels": data["labels"], "attention_mask": data["attention_mask"]}


def _load_id2label(run_name: str, model_dir_base: str) -> Dict[int, str]:
    model_dir = Path(model_dir_base) / run_name
    for info_path in model_dir.glob("fold_*/model_info.json"):
        data = json.loads(info_path.read_text())
        return {int(k): v for k, v in data["id2label"].items()}
    raise FileNotFoundError(f"No model_info.json found under {model_dir}")


def ensemble_folds(
    run_name: str,
    folds: Optional[Iterable[int]] = None,
    temperature: float = 1.0,
    output_dir_base: str = "outputs",
    allow_oof_concat: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average probabilities across folds for a single variant run."""
    if folds is None:
        preds_dir = Path(output_dir_base) / run_name / "preds"
        folds = sorted(int(p.stem.split("_")[-1]) for p in preds_dir.glob("fold_*.npz"))

    probs_list = []
    labels = None
    attention = None
    labels_list = []
    attention_list = []
    for fold in folds:
        data = _load_fold_logits(run_name, fold, output_dir_base)
        probs_list.append(_softmax(data["logits"], temperature))
        labels_list.append(data["labels"])
        attention_list.append(data["attention_mask"])
        if labels is None:
            labels = data["labels"]
            attention = data["attention_mask"]

    try:
        probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        return probs, labels, attention
    except ValueError:
        if not allow_oof_concat:
            raise
        probs = np.concatenate(probs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        attention = np.concatenate(attention_list, axis=0)
        return probs, labels, attention


def ensemble_variants(
    runs: Iterable[str],
    ensemble_name: str,
    temperature: float = 1.0,
    output_dir_base: str = "outputs",
    model_dir_base: str = "models",
) -> Dict[str, float]:
    """Average probabilities across multiple variant ensembles and save outputs."""
    runs = list(runs)
    run_probs = []
    labels = None
    attention = None
    for run_name in runs:
        probs, run_labels, run_attention = ensemble_folds(
            run_name,
            temperature=temperature,
            output_dir_base=output_dir_base,
        )
        run_probs.append(probs)
        if labels is None:
            labels = run_labels
            attention = run_attention

    probs = np.mean(np.stack(run_probs, axis=0), axis=0)
    id2label = _load_id2label(runs[0], model_dir_base)
    metrics = compute_metrics((probs, labels), id2label)

    out_dir = Path(output_dir_base) / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "ensemble_probs.npz", probs=probs, labels=labels, attention_mask=attention)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
