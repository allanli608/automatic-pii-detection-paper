"""Run analysis tasks across training runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.collect_runs import collect_runs
from src.analysis.compare_variants import compare_variants
from src.analysis.confusion import build_confusion_matrix
from src.analysis.plots import plot_confidence_histogram, plot_metric_curves


def _discover_runs(output_dir: str = "outputs") -> list[str]:
    runs = []
    for path in Path(output_dir).iterdir():
        if path.is_dir() and (path / "summary.json").exists():
            runs.append(path.name)
    return sorted(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--metric", default="fbeta")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = "outputs/analysis"
    runs = args.runs or _discover_runs()

    df = collect_runs()
    summary_path = Path(output_dir) / "runs_summary.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)

    for run_name in runs:
        plot_metric_curves(run_name, metric_key=f"eval_{args.metric}", output_dir=output_dir)
        plot_confidence_histogram(run_name, output_dir=output_dir)

        confusion = build_confusion_matrix(run_name)
        conf_path = Path(output_dir) / f"{run_name}_confusion.json"
        conf_path.write_text(json.dumps(confusion, indent=2))

    compare_variants(runs, metric=args.metric, output_dir=output_dir)


if __name__ == "__main__":
    main()
