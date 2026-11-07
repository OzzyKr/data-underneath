"""Typed configuration loaded from YAML. Every key must be present; unknown keys are rejected."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class ScaleConfig:
    n_windows: int
    window_ft: float
    window_px_fallback: int
    min_text_px: int
    min_points: int
    ransac_residual_ft: float
    max_slope_disagreement: float
    piecewise_break_ft: float
    max_depth_ft: float


@dataclass(frozen=True)
class SegmentConfig:
    core_width_fraction: float
    block_ft: float
    min_block_px: int
    min_thickness_ft: float
    penalty: float
    refine_window_ft: float
    colour_block_ft: float
    colour_min_block_px: int
    colour_min_thickness_ft: float
    colour_sliver_ft: float
    colour_penalty: float
    ink_luma: int
    blank_ink_fraction: float
    blank_fill_luma: int


@dataclass(frozen=True)
class ClusterConfig:
    distance_threshold: float
    colour_weight: float


@dataclass(frozen=True)
class LegendConfig:
    min_label_chars: int
    max_swatch_gap_frac: float
    min_swatch_px: int
    ocr_min_conf: int


@dataclass(frozen=True)
class ChecksConfig:
    min_thickness_ft: float
    depth_monotonic: bool


@dataclass(frozen=True)
class RenderConfig:
    max_height_px: int
    coverage: float


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int


@dataclass(frozen=True)
class AIConfig:
    tier: int
    provider: str


@dataclass(frozen=True)
class Config:
    scale: ScaleConfig
    segment: SegmentConfig
    cluster: ClusterConfig
    legend: LegendConfig
    checks: ChecksConfig
    render: RenderConfig
    runtime: RuntimeConfig
    ai: AIConfig

    def hash(self) -> str:
        """Short stable hash of every value, recorded with each result."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any], path: str) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"config section '{path}' must be a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    missing = names - set(data)
    if unknown:
        raise ValueError(f"unknown config keys in '{path}': {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing config keys in '{path}': {sorted(missing)}")
    kwargs = {}
    for name in names:
        typ = hints[name]
        value = data[name]
        if is_dataclass(typ):
            kwargs[name] = _build(typ, value, f"{path}.{name}" if path else name)  # type: ignore[arg-type]
        elif typ is float:
            kwargs[name] = float(value)
        elif typ is int:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"config '{path}.{name}' must be an integer, got {value!r}")
            kwargs[name] = int(value)
        elif typ is bool:
            if not isinstance(value, bool):
                raise ValueError(f"config '{path}.{name}' must be true/false, got {value!r}")
            kwargs[name] = value
        else:
            kwargs[name] = str(value)
    return cls(**kwargs)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the default config, then an optional override file, then optional in-code overrides."""
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        data = _merge(data, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        data = _merge(data, overrides)
    return _build(Config, data, "")
