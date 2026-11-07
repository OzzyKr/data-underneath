"""Split the lithology column into intervals where its fill (colour + pattern) changes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from skimage.color import rgb2lab

from lithology.config import SegmentConfig
from lithology.features import FeatureScaler, RowStats, describe_ranges, row_stats

log = logging.getLogger(__name__)


@dataclass
class Segment:
    top: int  # row within the analysed crop
    bottom: int  # exclusive
    blank: bool


@dataclass
class Segmentation:
    segments: list[Segment]
    stats: RowStats
    scaler: FeatureScaler
    block_px: int


def core_box_x(x0: int, x1: int, fraction: float) -> tuple[int, int]:
    """Central ``fraction`` of the column width (away from borders and printed text)."""
    w = x1 - x0
    margin = int(round(w * (1 - fraction) / 2))
    a, b = x0 + margin, x1 - margin
    return (a, b) if b - a >= 4 else (x0, x1)


def _blank_L(luma: int) -> float:
    return float(rgb2lab(np.full((1, 1, 3), luma / 255.0))[0, 0, 0])


def _refine(stats: RowStats, scaler: FeatureScaler, row: int, radius: int, w: int) -> int:
    """Move a boundary to the row that best separates the windows just above and just below it."""
    lo = max(w, row - radius)
    hi = min(stats.n_rows - w, row + radius)
    if hi <= lo:
        return row
    cand = np.arange(lo, hi + 1)
    above = scaler.transform(describe_ranges(stats, cand - w, cand))
    below = scaler.transform(describe_ranges(stats, cand, cand + w))
    score = np.linalg.norm(above - below, axis=1)
    return int(cand[int(np.argmax(score))])


def _absorb_thin(stats: RowStats, scaler: FeatureScaler, rows: list[int], min_px: int) -> list[int]:
    """Merge segments thinner than ``min_px`` into the more similar neighbour (thinnest first)."""
    rows = list(rows)
    while len(rows) > 2:
        heights = np.diff(rows)
        i = int(np.argmin(heights))
        if heights[i] >= min_px:
            break
        if i == 0:
            drop = 1  # first segment: remove the boundary below it
        elif i == len(heights) - 1:
            drop = i  # last segment: remove the boundary above it
        else:
            d = scaler.transform(describe_ranges(stats, np.array(rows[i - 1 : i + 2]), np.array(rows[i : i + 3])))
            up, down = np.linalg.norm(d[1] - d[0]), np.linalg.norm(d[1] - d[2])
            drop = i if up <= down else i + 1
        del rows[drop]
    return rows


def segment_column(rgb: np.ndarray, px_per_ft: float, cfg: SegmentConfig) -> Segmentation:
    """Segment a column crop (H, W, 3). Rows in the result are relative to the crop."""
    stats = row_stats(rgb, cfg.ink_luma)
    h = stats.n_rows
    block = max(cfg.min_block_px, int(round(cfg.block_ft * px_per_ft)))
    n_blocks = max(1, h // block)
    starts = np.arange(n_blocks) * block
    ends = np.append(starts[1:], h)
    desc = describe_ranges(stats, starts, ends)
    scaler = FeatureScaler.from_blocks(desc)
    X = scaler.transform(desc)

    min_size = max(1, int(round(cfg.min_thickness_ft * px_per_ft / block)))
    if n_blocks >= 2 * min_size + 1:
        pen = cfg.penalty * X.shape[1] * math.log(n_blocks)
        bkps = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(X).predict(pen=pen)
    else:
        bkps = [n_blocks]
    rows = [0]
    radius = max(block, int(round(cfg.refine_window_ft * px_per_ft)))
    for b in bkps[:-1]:
        r = _refine(stats, scaler, int(starts[b]), radius, block)
        if r - rows[-1] >= max(2, block // 2):
            rows.append(r)
    rows.append(h)

    rows = _absorb_thin(stats, scaler, rows, min_px=int(round(cfg.min_thickness_ft * px_per_ft)))

    # A segment is blank paper when most of its blocks are (robust to a stray border line or label).
    blank_L = _blank_L(cfg.blank_fill_luma)
    block_blank = (desc[:, 6] < cfg.blank_ink_fraction) & (desc[:, 0] > blank_L)
    segments = []
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        inside = (starts >= a) & (ends <= b)
        frac = float(block_blank[inside].mean()) if inside.any() else 0.0
        segments.append(Segment(top=a, bottom=b, blank=frac >= 0.5))
    log.debug("segmented %d rows into %d segments (block %d px)", h, len(segments), block)
    return Segmentation(segments=segments, stats=stats, scaler=scaler, block_px=block)


def interval_descriptors(seg: Segmentation, segments: list[Segment]) -> np.ndarray:
    """Descriptor per segment, trimmed by half a block at each end to avoid boundary contamination."""
    trim = seg.block_px // 2
    a = np.array([s.top + (trim if s.bottom - s.top > 3 * trim else 0) for s in segments])
    b = np.array([s.bottom - (trim if s.bottom - s.top > 3 * trim else 0) for s in segments])
    return describe_ranges(seg.stats, a, b)
