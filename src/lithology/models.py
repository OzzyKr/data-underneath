"""Typed data contracts shared by every pipeline step.

All coordinates are full-image pixels, origin top-left, with exclusive x1/y1 (the organisers' convention),
so ``image.crop(box.as_tuple())`` returns exactly the boxed region.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Output CSV columns, in the order of the organisers' example.csv.
CSV_COLUMNS: tuple[str, ...] = ("top_depth", "bottom_depth", "lithology", "top_y", "bottom_y", "cluster_id")

BOX_KEYS = ("depth_column", "interpreted_lithology_column", "lithology_legend")


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in full-image pixels, x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"empty or inverted box: {self}")

    @classmethod
    def from_seq(cls, seq: Any) -> Box:
        x0, y0, x1, y1 = (int(v) for v in seq)
        return cls(x0, y0, x1, y1)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def iou(self, other: Box) -> float:
        ix = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def clip(self, width: int, height: int) -> Box:
        return Box(max(0, self.x0), max(0, self.y0), min(width, self.x1), min(height, self.y1))


@dataclass(frozen=True)
class DocumentBoxes:
    """The three regions the organisers annotate (or the bonus detector predicts)."""

    depth_column: Box
    lithology_column: Box
    legend: Box

    @classmethod
    def from_dict(cls, boxes: dict[str, Any]) -> DocumentBoxes:
        missing = [k for k in BOX_KEYS if k not in boxes]
        if missing:
            raise ValueError(f"annotation is missing boxes: {missing}")
        return cls(
            depth_column=Box.from_seq(boxes["depth_column"]),
            lithology_column=Box.from_seq(boxes["interpreted_lithology_column"]),
            legend=Box.from_seq(boxes["lithology_legend"]),
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "depth_column": list(self.depth_column.as_tuple()),
            "interpreted_lithology_column": list(self.lithology_column.as_tuple()),
            "lithology_legend": list(self.legend.as_tuple()),
        }


@dataclass(frozen=True)
class Annotation:
    image_name: str
    width: int
    height: int
    boxes: DocumentBoxes

    @classmethod
    def from_json(cls, path: Path) -> Annotation:
        data = json.loads(Path(path).read_text())
        return cls(
            image_name=data["image_name"],
            width=int(data["width"]),
            height=int(data["height"]),
            boxes=DocumentBoxes.from_dict(data["bounding_boxes"]),
        )


@dataclass(frozen=True)
class DepthScale:
    """Pixel-row to depth mapping, piecewise linear through knots, extrapolated linearly at both ends.

    A plain linear scale has two knots. Logs with breaks or scale changes get more knots, so the mapping
    stays exact at every trusted depth label instead of averaging across a discontinuity.
    """

    knots_y: tuple[float, ...]
    knots_depth: tuple[float, ...]
    method: str  # "ocr-linear", "ocr-piecewise", "filename-dpi", ...
    n_points: int = 0
    residual_ft: float = 0.0  # median absolute residual of the fit at the OCR labels, in ft
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.knots_y) < 2 or len(self.knots_y) != len(self.knots_depth):
            raise ValueError("DepthScale needs at least two (y, depth) knots of equal length")
        if any(b <= a for a, b in zip(self.knots_y, self.knots_y[1:], strict=False)):
            raise ValueError("DepthScale knots_y must be strictly increasing")

    @classmethod
    def linear(cls, ft_per_px: float, depth_at_y0: float, method: str, **kw: Any) -> DepthScale:
        return cls((0.0, 1000.0), (depth_at_y0, depth_at_y0 + 1000.0 * ft_per_px), method, **kw)

    @property
    def ft_per_px(self) -> float:
        """Overall slope from the first to the last knot."""
        return (self.knots_depth[-1] - self.knots_depth[0]) / (self.knots_y[-1] - self.knots_y[0])

    @property
    def px_per_ft(self) -> float:
        return 1.0 / self.ft_per_px

    def depth(self, y: float) -> float:
        ys, ds = self.knots_y, self.knots_depth
        if y <= ys[0]:
            i = 0
        elif y >= ys[-1]:
            i = len(ys) - 2
        else:
            i = max(j for j in range(len(ys) - 1) if ys[j] <= y)
        slope = (ds[i + 1] - ds[i]) / (ys[i + 1] - ys[i])
        return ds[i] + slope * (y - ys[i])


@dataclass
class LegendEntry:
    """One swatch-plus-label pair read from the legend."""

    label: str
    swatch: Box  # full-image pixels
    ocr_confidence: float = 0.0
    is_rock: bool = True  # False for drilling/show symbols and modifier words (never used to name intervals)
    rgb: tuple[int, int, int] | None = None  # swatch fill colour (median of non-ink pixels)
    chroma: float = 0.0  # max - min of ``rgb``: ~0 for black-and-white legends


@dataclass
class Interval:
    """One lithology interval: a row of the output CSV plus diagnostics."""

    top_y: int
    bottom_y: int
    lithology: str = "Unknown"
    cluster_id: int | None = None
    top_depth: float = float("nan")
    bottom_depth: float = float("nan")
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)
    status: str = "named"  # named | similar (2-3 look-alike legend entries) | unresolved (nothing matches clearly)
    alternatives: list[str] = field(default_factory=list)  # the look-alike or candidate entries, best guess first

    def __post_init__(self) -> None:
        if self.bottom_y <= self.top_y:
            raise ValueError(f"interval must have bottom_y > top_y, got {self.top_y}..{self.bottom_y}")

    @property
    def height_px(self) -> int:
        return self.bottom_y - self.top_y

    def csv_row(self) -> dict[str, Any]:
        return {
            "top_depth": round(self.top_depth, 2),
            "bottom_depth": round(self.bottom_depth, 2),
            "lithology": self.lithology,
            "top_y": self.top_y,
            "bottom_y": self.bottom_y,
            "cluster_id": "" if self.cluster_id is None else self.cluster_id,
        }


@dataclass
class DocumentResult:
    """Everything the pipeline produced for one document, including how it went."""

    doc_id: str
    intervals: list[Interval]
    boxes: DocumentBoxes | None = None
    scale: DepthScale | None = None
    legend: list[LegendEntry] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    naming: list[dict[str, Any]] = field(default_factory=list)  # per rock group: both opinions + the decision
    layout: dict[str, Any] = field(default_factory=dict)  # set when the boxes were detected (bonus), not annotated

    @property
    def ok(self) -> bool:
        return self.error is None
