"""Run a single training variant with status tracking."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from src.training.train_kfold import train_kfold
from src.training.variants import load_variants


def _project_root() -> Path:
    return PROJECT_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _configure_hf_mirror() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    if not os.environ.get("HF_HUB_DISABLE_XET"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _status_path(run_name: str, output_dir_base: str) -> Path:
    out_dir = Path(output_dir_base) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "status.json"


def read_status(run_name: str, output_dir_base: str) -> Dict[str, Any]:
    path = _status_path(run_name, output_dir_base)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def write_status(run_name: str, output_dir_base: str, status: Dict[str, Any]) -> None:
    path = _status_path(run_name, output_dir_base)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, path)


def init_status(run_name: str, output_dir_base: str, overrides: Dict[str, Any], args) -> Dict[str, Any]:
    status = {
        "run_name": run_name,
        "variant": args.variant,
        "seed": args.seed,
        "git_commit": get_git_commit(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "start_time_utc": _utc_now_iso(),
        "end_time_utc": None,
        "status": "running",
        "overrides": overrides,
        "folds": {},
        "error": None,
    }
    write_status(run_name, output_dir_base, status)
    return status


def update_fold_status(run_name: str, output_dir_base: str, fold: int, fold_status: str, extra=None) -> None:
    status = read_status(run_name, output_dir_base)
    status.setdefault("folds", {})
    entry = status["folds"].get(str(fold), {})
    entry.update({
        "status": fold_status,
        "updated_time_utc": _utc_now_iso(),
    })
    if extra:
        entry.update(extra)
    status["folds"][str(fold)] = entry
    write_status(run_name, output_dir_base, status)


def finalize_status_success(run_name: str, output_dir_base: str) -> None:
    status = read_status(run_name, output_dir_base)
    status["status"] = "success"
    status["end_time_utc"] = _utc_now_iso()
    write_status(run_name, output_dir_base, status)


def finalize_status_failure(run_name: str, output_dir_base: str, exc: BaseException) -> None:
    status = read_status(run_name, output_dir_base)
    status["status"] = "failed"
    status["end_time_utc"] = _utc_now_iso()
    status["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_status(run_name, output_dir_base, status)


def apply_variant(config, variant_spec, run_name: str, seed: int) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "RUN_NAME": run_name,
        "SEED": seed,
        "WEIGHT_MODE": variant_spec.weight_mode,
        "MODEL_MODE": variant_spec.model_mode,
    }

    if variant_spec.learning_rate is not None:
        overrides["LEARNING_RATE"] = variant_spec.learning_rate
    if variant_spec.weight_decay is not None:
        overrides["WEIGHT_DECAY"] = variant_spec.weight_decay
    if variant_spec.external_fraction is not None:
        overrides["EXTERNAL_FRACTION"] = variant_spec.external_fraction
    if variant_spec.o_weight is not None:
        overrides["O_WEIGHT"] = variant_spec.o_weight
    if variant_spec.md_k is not None:
        overrides["MD_K"] = variant_spec.md_k
    if variant_spec.md_p is not None:
        overrides["MD_P"] = variant_spec.md_p

    for key, value in overrides.items():
        setattr(config, key, value)

    return overrides


def save_run_metadata(config, variant_spec, overrides: Dict[str, Any]) -> None:
    out_dir = Path(config.OUTPUT_DIR_BASE) / config.RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": config.RUN_NAME,
        "variant": variant_spec.name,
        "git_commit": get_git_commit(),
        "overrides": overrides,
        "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=_json_default))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="Variant name from config/variants.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--variants-file", default="config/variants.yaml")
    args = parser.parse_args()

    os.chdir(_project_root())

    _configure_hf_mirror()

    config = load_config()
    variants_file = Path(args.variants_file)
    if not variants_file.is_absolute():
        variants_file = PROJECT_ROOT / variants_file

    variants = load_variants(variants_file)
    if args.variant not in variants:
        raise ValueError(f"Unknown variant: {args.variant}")

    variant_spec = variants[args.variant]
    run_name = args.run_name or variant_spec.name
    seed = args.seed if args.seed is not None else config.SEED

    overrides = apply_variant(config, variant_spec, run_name=run_name, seed=seed)

    set_seed(config.SEED)

    save_run_metadata(config, variant_spec, overrides)
    init_status(config.RUN_NAME, config.OUTPUT_DIR_BASE, overrides, args)

    def status_hook(fold: int, fold_status: str, extra=None) -> None:
        update_fold_status(config.RUN_NAME, config.OUTPUT_DIR_BASE, fold, fold_status, extra)

    try:
        train_kfold(config, variant_spec, status_hook=status_hook)
        finalize_status_success(config.RUN_NAME, config.OUTPUT_DIR_BASE)

        done_path = Path(config.MODEL_DIR_BASE) / config.RUN_NAME / "DONE"
        done_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.write_text("Done variant\n")
    except Exception as exc:
        finalize_status_failure(config.RUN_NAME, config.OUTPUT_DIR_BASE, exc)
        raise


if __name__ == "__main__":
    main()
