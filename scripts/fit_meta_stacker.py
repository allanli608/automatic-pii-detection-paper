"""Train a meta stacker on out-of-fold predictions."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.stacker import train_stacker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runs", nargs="+", required=True, help="Run names to stack")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    run_name = args.run_name or f"stacker_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path("outputs") / run_name
    train_stacker(
        runs=args.base_runs,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
