"""Utilities to compare variants and export summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from src.analysis.plots import plot_variant_comparison


def compare_variants(
    run_names: Iterable[str],
    metric: str = "fbeta",
    output_dir: str = "outputs/analysis",
) -> Path:
    """Create a comparison report JSON and a plot for variant runs."""
    run_names = list(run_names)
    report: List[dict] = []

    metric_key = metric if metric.startswith("eval_") else f"eval_{metric}"
    for run_name in run_names:
        summary_path = Path("outputs") / run_name / "summary.json"
        data = json.loads(summary_path.read_text())
        stats = data.get("summary", {}).get(metric_key, {})
        report.append({
            "run_name": run_name,
            "metric": metric,
            "mean": stats.get("mean"),
            "std": stats.get("std"),
        })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "compare_variants.json"
    report_path.write_text(json.dumps(report, indent=2))

    plot_variant_comparison(run_names, metric=metric, output_dir=output_dir)
    return report_path
