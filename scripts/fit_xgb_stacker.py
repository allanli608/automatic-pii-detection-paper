"""Train an XGBoost stacker on out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.xgb_stacker import train_xgb_stacker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runs", nargs="+", required=True, help="Run names to stack")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--params", default=None, help="JSON string of XGBoost params override")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    run_name = args.run_name or f"xgb_stacker_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path("outputs") / run_name

    params = json.loads(args.params) if args.params else None
    train_xgb_stacker(
        runs=args.base_runs,
        output_dir=str(output_dir),
        params=params,
        num_boost_round=args.num_boost_round,
    )


if __name__ == "__main__":
    main()
