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

from src.inference.ensemble import ensemble_folds, ensemble_variants
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


def eval_mlp_stacker(base_runs, stacker_path: str | None, output_dir: str) -> Dict[str, float]:
    if stacker_path:
        model = load_stacker(stacker_path)
    else:
        model_path = train_stacker(base_runs, output_dir=str(output_dir))
        model = load_stacker(model_path)

    probs, labels = predict_with_stacker(model, base_runs)
    return compute_metrics((probs, labels), _load_id2label(base_runs[0]))


def eval_xgb_stacker(base_runs, model_path: str | None, output_dir: str) -> Dict[str, float]:
    if model_path:
        booster = load_xgb_stacker(model_path)
    else:
        model_path = train_xgb_stacker(base_runs, output_dir=str(output_dir))
        booster = load_xgb_stacker(model_path)

    probs, labels = predict_with_xgb_stacker(booster, base_runs)
    return compute_metrics((probs, labels), _load_id2label(base_runs[0]))


def eval_variant_ensemble(runs, temperature: float, ensemble_name: str) -> Dict[str, float]:
    return ensemble_variants(runs, ensemble_name=ensemble_name, temperature=temperature)


def _discover_runs(output_dir: str = "outputs") -> list[str]:
    runs = []
    for path in Path(output_dir).iterdir():
        if path.is_dir() and (path / "summary.json").exists():
            runs.append(path.name)
    return sorted(runs)


def _summarize_metrics(run_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not run_metrics:
        return {}
    metric_keys = list(next(iter(run_metrics.values())).keys())
    totals = {key: 0.0 for key in metric_keys}
    for metrics in run_metrics.values():
        for key in metric_keys:
            totals[key] += metrics[key]
    mean = {key: totals[key] / len(run_metrics) for key in metric_keys}
    max_vals = {key: max(m[key] for m in run_metrics.values()) for key in metric_keys}
    return {"mean": mean, "max": max_vals}


def _best_run_by_metric(run_metrics: Dict[str, Dict[str, float]], metric_key: str) -> Dict[str, object]:
    if not run_metrics:
        return {}
    best_run, best_metrics = max(
        run_metrics.items(),
        key=lambda item: item[1].get(metric_key, float("-inf")),
    )
    return {"run": best_run, "metrics": best_metrics}


def _rank_methods(method_scores: Dict[str, float]) -> list[Dict[str, float | str]]:
    ranked = sorted(method_scores.items(), key=lambda item: item[1], reverse=True)
    return [{"method": name, "score": score} for name, score in ranked]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run name for fold ensemble")
    parser.add_argument("--base-runs", nargs="+", default=None, help="Runs to use for stackers")
    parser.add_argument("--runs", nargs="*", default=None, help="Run names to evaluate across methods")
    parser.add_argument("--all-runs", action="store_true", help="Use all runs discovered in outputs/")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mlp-stacker", default=None, help="Path to stacker.pt")
    parser.add_argument("--mlp-stacker-out", default=None, help="Output dir for training MLP stacker")
    parser.add_argument("--xgb-stacker", default=None, help="Path to xgb_stacker.json")
    parser.add_argument("--xgb-stacker-out", default=None, help="Output dir for training XGB stacker")
    parser.add_argument("--ensemble-name", default=None, help="Name for cross-run ensemble outputs")
    parser.add_argument("--metric", default="fbeta", help="Metric key used to select the best method")
    parser.add_argument("--out", default="outputs/analysis/methods.json")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    runs = None
    if args.all_runs:
        runs = _discover_runs()
    elif args.runs:
        runs = args.runs

    results: Dict[str, object] = {}
    if runs:
        ensemble_name = args.ensemble_name or "ensemble_all_runs"
        mlp_output_dir = args.mlp_stacker_out or "outputs/stacker_all_runs"
        xgb_output_dir = args.xgb_stacker_out or "outputs/xgb_stacker_all_runs"

        fold_metrics = {run: eval_ensemble(run, temperature=args.temperature) for run in runs}
        fold_summary = _summarize_metrics(fold_metrics)
        results["runs"] = runs
        results["fold_ensemble"] = {
            "by_run": fold_metrics,
            "summary": fold_summary,
            "best_run_by_metric": _best_run_by_metric(fold_metrics, args.metric),
        }
        results["variant_ensemble"] = eval_variant_ensemble(
            runs,
            temperature=args.temperature,
            ensemble_name=ensemble_name,
        )
        results["mlp_stacker"] = eval_mlp_stacker(runs, args.mlp_stacker, mlp_output_dir)
        try:
            results["xgb_stacker"] = eval_xgb_stacker(runs, args.xgb_stacker, xgb_output_dir)
        except ImportError as exc:
            results["xgb_stacker"] = {"error": str(exc)}

        method_scores: Dict[str, float] = {}
        if fold_summary:
            method_scores["fold_ensemble_mean"] = fold_summary["mean"].get(args.metric, float("-inf"))
        for key in ("variant_ensemble", "mlp_stacker", "xgb_stacker"):
            metrics = results.get(key, {})
            if isinstance(metrics, dict) and args.metric in metrics:
                method_scores[key] = metrics[args.metric]

        ranking = _rank_methods(method_scores)
        results["method_ranking"] = ranking
        if ranking:
            results["best_method"] = {"metric": args.metric, **ranking[0]}
    else:
        if not args.run or not args.base_runs:
            parser.error("Provide --run/--base-runs or use --runs/--all-runs.")
        results = {
            "ensemble": eval_ensemble(args.run, temperature=args.temperature),
            "mlp_stacker": eval_mlp_stacker(args.base_runs, args.mlp_stacker, "outputs/stacker_temp"),
        }
        try:
            results["xgb_stacker"] = eval_xgb_stacker(args.base_runs, args.xgb_stacker, "outputs/xgb_stacker_temp")
        except ImportError as exc:
            results["xgb_stacker"] = {"error": str(exc)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
