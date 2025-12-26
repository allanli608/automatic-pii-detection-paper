"""Evaluate ensemble, MLP stacker, and XGBoost stacker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.ensemble import ensemble_folds
from src.inference.stacker import load_stacker, predict_with_stacker, train_stacker
from src.inference.xgb_stacker import load_xgb_stacker, predict_with_xgb_stacker, train_xgb_stacker
from src.metrics.metrics import compute_metrics


def _load_id2label(run_name: str) -> Dict[int, str]:
    model_dir = Path("models") / run_name
    for info_path in model_dir.glob("fold_*/model_info.json"):
        data = json.loads(info_path.read_text())
        return {int(k): v for k, v in data["id2label"].items()}
    raise FileNotFoundError(f"No model_info.json found under {model_dir}")


def eval_ensemble(run_name: str, temperature: float) -> Dict[str, float]:
    probs, labels, _ = ensemble_folds(run_name, temperature=temperature)
    return compute_metrics((probs, labels), _load_id2label(run_name))


def eval_mlp_stacker(base_runs, stacker_path: str | None) -> Dict[str, float]:
    if stacker_path:
        model = load_stacker(stacker_path)
    else:
        output_dir = Path("outputs") / "stacker_temp"
        model_path = train_stacker(base_runs, output_dir=str(output_dir))
        model = load_stacker(model_path)

    probs, labels = predict_with_stacker(model, base_runs)
    return compute_metrics((probs, labels), _load_id2label(base_runs[0]))


def eval_xgb_stacker(base_runs, model_path: str | None) -> Dict[str, float]:
    if model_path:
        booster = load_xgb_stacker(model_path)
    else:
        output_dir = Path("outputs") / "xgb_stacker_temp"
        model_path = train_xgb_stacker(base_runs, output_dir=str(output_dir))
        booster = load_xgb_stacker(model_path)

    probs, labels = predict_with_xgb_stacker(booster, base_runs)
    return compute_metrics((probs, labels), _load_id2label(base_runs[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name for fold ensemble")
    parser.add_argument("--base-runs", nargs="+", required=True, help="Runs to use for stackers")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mlp-stacker", default=None, help="Path to stacker.pt")
    parser.add_argument("--xgb-stacker", default=None, help="Path to xgb_stacker.json")
    parser.add_argument("--out", default="outputs/analysis/methods.json")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    results = {
        "ensemble": eval_ensemble(args.run, temperature=args.temperature),
        "mlp_stacker": eval_mlp_stacker(args.base_runs, args.mlp_stacker),
        "xgb_stacker": eval_xgb_stacker(args.base_runs, args.xgb_stacker),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
