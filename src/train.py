import argparse
import os
import json
import subprocess
import random
import numpy as np
import torch
import config

## Helpers for logging status. Can move later.
import time
import socket
import traceback
from datetime import datetime, timezone

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

## Helpers for logging status. Can move later.

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _status_path():
    out_dir = os.path.join(config.OUTPUT_DIR_BASE, config.RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "status.json")

def read_status():
    path = _status_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def write_status(status: dict):
    path = _status_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)

def init_status(overrides: dict, args):
    status = {
        "run_name": config.RUN_NAME,
        "variant": getattr(args, "variant", None),
        "seed": getattr(args, "seed", None),
        "git_commit": get_git_commit(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "start_time_utc": _utc_now_iso(),
        "end_time_utc": None,
        "status": "running",   # running|success|failed
        "overrides": overrides,
        "folds": {},           # filled as folds run
        "error": None,         # filled on failure
    }
    write_status(status)
    return status

def update_fold_status(fold: int, fold_status: str, extra: dict | None = None):
    status = read_status()
    status.setdefault("folds", {})
    entry = status["folds"].get(str(fold), {})
    entry.update({
        "status": fold_status,   # running|success|failed
        "updated_time_utc": _utc_now_iso(),
    })
    if extra:
        entry.update(extra)
    status["folds"][str(fold)] = entry
    write_status(status)

def finalize_status_success():
    status = read_status()
    status["status"] = "success"
    status["end_time_utc"] = _utc_now_iso()
    write_status(status)

def finalize_status_failure(exc: BaseException):
    status = read_status()
    status["status"] = "failed"
    status["end_time_utc"] = _utc_now_iso()
    status["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_status(status)