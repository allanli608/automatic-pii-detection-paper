"""Configuration loader and helpers."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Dict, Optional

import torch


def _module_to_dict(module) -> Dict[str, object]:
    return {name: getattr(module, name) for name in dir(module) if name.isupper()}


def _ensure_bf16_support(cfg: SimpleNamespace) -> None:
    if getattr(cfg, "BF16", False):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            cfg.BF16 = False
            if not getattr(cfg, "FP16", False):
                cfg.FP16 = True


def load_config(
    base_module: str = "config.base",
    override_module: Optional[str] = "config.gpu_5090",
    extra: Optional[Dict[str, object]] = None,
) -> SimpleNamespace:
    """Load config modules into a SimpleNamespace."""
    base = _module_to_dict(import_module(base_module))
    overrides = _module_to_dict(import_module(override_module)) if override_module else {}
    data = {**base, **overrides}
    if extra:
        data.update(extra)

    cfg = SimpleNamespace(**data)
    _ensure_bf16_support(cfg)
    return cfg
