"""Evaluate fold ensembles or stacker outputs."""

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
from src.inference.stacker import load_stacker, predict_with_stacker
from src.metrics.metrics import compute_metrics


def _load_id2label(run_name: str) -> Dict[int, str]:
    model_dir = Path("models") / run_name
    for info_path in model_dir.glob("fold_*/model_info.json"):
        data = json.loads(info_path.read_text())
        return {int(k): v for k, v in data["id2label"].items()}
    raise FileNotFoundError(f"No model_info.json found under {model_dir}")


def eval_fold_ensemble(run_name: str, temperature: float) -> Dict[str, float]:
    probs, labels, _ = ensemble_folds(run_name, temperature=temperature)
    metrics = compute_metrics((probs, labels), _load_id2label(run_name))
    out_path = Path("outputs") / run_name / "ensemble_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def eval_stacker(stacker_path: str) -> Dict[str, float]:
    stacker_path = Path(stacker_path)
    config_path = stacker_path.parent / "stacker_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing stacker_config.json in {stacker_path.parent}")

    config = json.loads(config_path.read_text())
    base_runs = config["runs"]

    model = load_stacker(stacker_path)
    probs, labels = predict_with_stacker(model, base_runs)
    metrics = compute_metrics((probs, labels), _load_id2label(base_runs[0]))

    out_path = stacker_path.parent / "stacker_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run name to evaluate fold ensemble")
    parser.add_argument("--stacker", default=None, help="Path to stacker.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if not args.run and not args.stacker:
        parser.error("Provide --run and/or --stacker for evaluation.")

    if args.run:
        metrics = eval_fold_ensemble(args.run, temperature=args.temperature)
        print(json.dumps(metrics, indent=2))

    if args.stacker:
        metrics = eval_stacker(args.stacker)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
