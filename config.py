"""Legacy config shim to load config/base.py and config/gpu_5090.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import runpy
import torch

_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load_module_dict(path: Path) -> Dict[str, object]:
    data = runpy.run_path(str(path))
    return {k: v for k, v in data.items() if k.isupper()}


def _ensure_bf16_support(cfg: SimpleNamespace) -> None:
    if getattr(cfg, "BF16", False):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            cfg.BF16 = False
            if not getattr(cfg, "FP16", False):
                cfg.FP16 = True


def load_config(
    base_path: Optional[Path] = None,
    override_path: Optional[Path] = None,
    extra: Optional[Dict[str, object]] = None,
) -> SimpleNamespace:
    """Load config values from base and override files."""
    base_path = base_path or (_CONFIG_DIR / "base.py")
    override_path = override_path or (_CONFIG_DIR / "gpu_5090.py")

    base = _load_module_dict(base_path)
    overrides = _load_module_dict(override_path) if override_path else {}

    data = {**base, **overrides}
    if extra:
        data.update(extra)

    cfg = SimpleNamespace(**data)
    _ensure_bf16_support(cfg)
    return cfg


# Load defaults into module namespace for backwards compatibility
_default = load_config()
for _key, _value in _default.__dict__.items():
    globals()[_key] = _value

__all__ = ["load_config"] + [k for k in globals() if k.isupper()]
