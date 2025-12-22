"""Collect summary metrics from all runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

def collect_runs(output_dir_base: str = "outputs"):
    """Collect summary.json files into a DataFrame."""
    run_dirs = Path(output_dir_base).glob("*/summary.json")
    rows: List[dict] = []

    for summary_path in run_dirs:
        data = json.loads(summary_path.read_text())
        run_name = summary_path.parent.name
        row = {"run_name": run_name}
        summary = data.get("summary", {})
        for metric, stats in summary.items():
            row[f"{metric}_mean"] = stats.get("mean")
            row[f"{metric}_std"] = stats.get("std")
        rows.append(row)

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for collect_runs().") from exc

    return pd.DataFrame(rows)
