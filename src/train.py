import argparse
import os
import json
import subprocess
import random
import numpy as np
import torch

import config
from src.train_baseline import train_baseline  # or import train_fold if you prefer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def apply_variant(variant: str, args):
    """
    Map a variant name to config overrides.
    You can extend this dictionary over time.
    """
    # defaults (baseline)
    overrides = {
        "RUN_NAME": variant,
        "EXTERNAL_FRACTION": 1.0,
        "SEED": args.seed,
        "O_WEIGHT": 0.05,
        "LEARNING_RATE": config.LEARNING_RATE,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
    }

    if variant == "baseline":
        overrides["RUN_NAME"] = "baseline"
    elif variant == "external_0":
        overrides["RUN_NAME"] = "external_0"
        overrides["EXTERNAL_FRACTION"] = 0.0
    elif variant == "external_50":
        overrides["RUN_NAME"] = "external_50"
        overrides["EXTERNAL_FRACTION"] = 0.5
    elif variant == "o_010":
        overrides["RUN_NAME"] = "o_010"
        overrides["O_WEIGHT"] = 0.10
    elif variant == "lr_half":
        overrides["RUN_NAME"] = "lr_half"
        overrides["LEARNING_RATE"] = config.LEARNING_RATE * 0.5
    elif variant == "wd_high":
        overrides["RUN_NAME"] = "wd_high"
        overrides["WEIGHT_DECAY"] = max(config.WEIGHT_DECAY * 2, config.WEIGHT_DECAY + 0.01)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Allow CLI flags to override even the variant defaults
    if args.external_fraction is not None:
        overrides["EXTERNAL_FRACTION"] = args.external_fraction
    if args.o_weight is not None:
        overrides["O_WEIGHT"] = args.o_weight
    if args.lr is not None:
        overrides["LEARNING_RATE"] = args.lr
    if args.weight_decay is not None:
        overrides["WEIGHT_DECAY"] = args.weight_decay
    if args.run_name is not None:
        overrides["RUN_NAME"] = args.run_name

    # Apply overrides to config module
    for k, v in overrides.items():
        setattr(config, k, v)

    return overrides


def save_run_metadata(overrides: dict):
    out_dir = os.path.join(config.OUTPUT_DIR_BASE, config.RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "run_name": config.RUN_NAME,
        "git_commit": get_git_commit(),
        "overrides": overrides,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        help="baseline|external_0|external_50|o_010|lr_half|wd_high")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)

    # optional direct knobs
    parser.add_argument("--external-fraction", type=float, default=None)
    parser.add_argument("--o-weight", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    overrides = apply_variant(args.variant, args)
    save_run_metadata(overrides)

    train_baseline(num_folds=config.NUM_FOLDS)


if __name__ == "__main__":
    main()
