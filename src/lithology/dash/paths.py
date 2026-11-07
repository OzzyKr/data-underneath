"""Where the dashboard reads and writes. Everything generated lives under ``dashboard/.cache`` (gitignored)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    repo: Path = REPO

    @property
    def out_batch(self) -> Path:
        return self.repo / "out"

    @property
    def out_live(self) -> Path:
        return self.repo / "out_live"

    @property
    def images(self) -> Path:
        return self.repo / "dataset" / "images"

    @property
    def annotations(self) -> Path:
        return self.repo / "dataset" / "bbox_annotation"

    @property
    def selection(self) -> Path:
        return self.repo / "labels" / "selection.yaml"

    @property
    def cache(self) -> Path:
        return self.repo / "dashboard" / ".cache"

    @property
    def db(self) -> Path:
        return self.cache / "logs.duckdb"

    @property
    def tiles(self) -> Path:
        return self.cache / "tiles"

    @property
    def uploads(self) -> Path:
        return self.cache / "uploads"

    @property
    def ai_cache(self) -> Path:
        return self.cache / "ai"

    @property
    def build_meta(self) -> Path:
        return self.cache / "build.json"

    @property
    def dist(self) -> Path:
        return self.repo / "dashboard" / "dist"
