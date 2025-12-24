"""Run all variants defined in config/variants.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.variants import load_variants

def _project_root() -> Path:
    return PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-file", default="config/variants.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    variants_file = Path(args.variants_file)
    if not variants_file.is_absolute():
        variants_file = PROJECT_ROOT / variants_file

    variants = load_variants(variants_file)
    variant_names = list(variants.keys())
    if args.variants:
        variant_names = [name for name in variant_names if name in set(args.variants)]

    root = _project_root()
    for name in variant_names:
        cmd = [sys.executable, str(root / "scripts" / "run_variant.py"), "--variant", name]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        subprocess.run(cmd, check=True, cwd=str(root))


if __name__ == "__main__":
    main()
