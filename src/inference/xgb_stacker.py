"""XGBoost-based stacker for token classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from src.inference.stacker import build_stacker_dataset


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for the XGB stacker.") from exc
    return xgb


def train_xgb_stacker(
    runs: Iterable[str],
    output_dir: str,
    params: Optional[dict] = None,
    num_boost_round: int = 200,
) -> Path:
    """Train an XGBoost stacker from out-of-fold predictions."""
    xgb = _require_xgboost()

    features, labels = build_stacker_dataset(runs)
    num_classes = int(labels.max()) + 1

    dtrain = xgb.DMatrix(features, label=labels)

    default_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    if params:
        default_params.update(params)

    booster = xgb.train(default_params, dtrain, num_boost_round=num_boost_round)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgb_stacker.json"
    booster.save_model(str(model_path))

    config = {
        "runs": list(runs),
        "params": default_params,
        "num_boost_round": num_boost_round,
        "num_classes": num_classes,
        "input_dim": int(features.shape[1]),
    }
    (out_dir / "xgb_stacker_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "feature_list.json").write_text(
        json.dumps({"runs": list(runs), "input_dim": int(features.shape[1])}, indent=2)
    )

    return model_path


def load_xgb_stacker(path: str | Path):
    """Load a saved XGBoost stacker."""
    xgb = _require_xgboost()
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def predict_with_xgb_stacker(
    booster,
    runs: Iterable[str],
    output_dir_base: str = "outputs",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict stacked probabilities using a trained XGBoost stacker."""
    xgb = _require_xgboost()
    features, labels = build_stacker_dataset(runs, output_dir_base=output_dir_base)
    dtest = xgb.DMatrix(features)
    probs = booster.predict(dtest)
    return probs, labels
