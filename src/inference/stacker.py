"""Stacking model for token classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _load_fold_preds(run_name: str, fold: int, output_dir_base: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(output_dir_base) / run_name / "preds" / f"fold_{fold}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    data = np.load(path)
    return data["logits"], data["labels"], data["attention_mask"]


def _collect_folds(runs: Iterable[str], output_dir_base: str) -> List[int]:
    first = next(iter(runs))
    preds_dir = Path(output_dir_base) / first / "preds"
    return sorted(int(p.stem.split("_")[-1]) for p in preds_dir.glob("fold_*.npz"))


def build_stacker_dataset(
    runs: Iterable[str],
    output_dir_base: str = "outputs",
    folds: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build stacking features and labels from out-of-fold predictions."""
    runs = list(runs)
    if folds is None:
        folds = _collect_folds(runs, output_dir_base)

    feature_rows = []
    label_rows = []

    for fold in folds:
        run_probs = []
        labels = None
        attention = None
        for run_name in runs:
            logits, run_labels, run_attention = _load_fold_preds(run_name, fold, output_dir_base)
            run_probs.append(_softmax(logits))
            if labels is None:
                labels = run_labels
                attention = run_attention

        stacked = np.concatenate(run_probs, axis=-1)
        mask = (labels != -100) & (attention == 1)
        feature_rows.append(stacked[mask])
        label_rows.append(labels[mask])

    features = np.concatenate(feature_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)

    return features, labels


class StackerMLP(nn.Module):
    """Simple MLP stacker for token probabilities."""

    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_stacker(
    runs: Iterable[str],
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    device: Optional[torch.device] = None,
) -> Path:
    """Train a stacking model from out-of-fold predictions."""
    features, labels = build_stacker_dataset(runs)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StackerMLP(input_dim=x.size(1), num_labels=int(y.max().item()) + 1, hidden_dim=hidden_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "stacker.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": x.size(1), "num_labels": int(y.max().item()) + 1, "hidden_dim": hidden_dim}, model_path)

    config = {
        "runs": list(runs),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
    }
    (out_dir / "stacker_config.json").write_text(json.dumps(config, indent=2))

    feature_list = {
        "runs": list(runs),
        "input_dim": x.size(1),
        "num_labels": int(y.max().item()) + 1,
    }
    (out_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

    return model_path


def load_stacker(path: str | Path, device: Optional[torch.device] = None) -> StackerMLP:
    """Load a saved stacker model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(path, map_location=device)
    model = StackerMLP(
        input_dim=data["input_dim"],
        num_labels=data["num_labels"],
        hidden_dim=data["hidden_dim"],
    )
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_with_stacker(
    model: StackerMLP,
    runs: Iterable[str],
    output_dir_base: str = "outputs",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict stacked probabilities using a trained stacker."""
    features, labels = build_stacker_dataset(runs, output_dir_base=output_dir_base)

    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32, device=device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    return probs, labels
