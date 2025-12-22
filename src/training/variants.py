"""Variant specifications and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class VariantSpec:
    """Variant configuration for a training run."""

    name: str
    weight_mode: str
    model_mode: str
    o_weight: Optional[float] = None
    md_k: Optional[int] = None
    md_p: Optional[float] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    external_fraction: Optional[float] = None


def load_variants(path: str | Path) -> Dict[str, VariantSpec]:
    """Load variant specs from a YAML file."""
    variant_path = Path(path)
    data = yaml.safe_load(variant_path.read_text()) or {}
    raw_variants = data.get("variants", data)

    variants: Dict[str, VariantSpec] = {}
    for name, spec in raw_variants.items():
        variants[name] = VariantSpec(
            name=name,
            weight_mode=spec.get("weight_mode", "uniform"),
            model_mode=spec.get("model_mode", "baseline"),
            o_weight=spec.get("o_weight"),
            md_k=spec.get("md_k"),
            md_p=spec.get("md_p"),
            learning_rate=spec.get("learning_rate"),
            weight_decay=spec.get("weight_decay"),
            external_fraction=spec.get("external_fraction"),
        )

    return variants
