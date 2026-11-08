"""Split the lithology column into intervals where its fill (colour + pattern) changes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from skimage.color import rgb2lab

from lithology.config import SegmentConfig
from lithology.features import DEFAULT_FLOOR, FILL_IDX, FeatureScaler, RowStats, describe_ranges, row_stats

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


def _refine(
    stats: RowStats, scaler: FeatureScaler, row: int, radius: int, w: int, idx: np.ndarray | None = None
) -> int:
    """Move a boundary to the row that best separates the windows just above and just below it.

    ``idx`` restricts the comparison to a feature subset (its scaler must match that subset).
    """
    lo = max(w, row - radius)
    hi = min(stats.n_rows - w, row + radius)
    if hi <= lo:
        return row
    cand = np.arange(lo, hi + 1)
    da = describe_ranges(stats, cand - w, cand)
    db = describe_ranges(stats, cand, cand + w)
    if idx is not None:
        da, db = da[:, idx], db[:, idx]
    above = scaler.transform(da)
    below = scaler.transform(db)
    score = np.linalg.norm(above - below, axis=1)
    return int(cand[int(np.argmax(score))])


def snap_to_gap(stats: RowStats, scaler: FeatureScaler, row: int, radius: int, w: int) -> int:
    """Move a pattern boundary into the blank gap between two patterns' marks.

    Only same-fill boundaries move (colour changes are already sharp). Among the gaps within ``radius``, the one
    that best separates the windows above and below wins.
    """
    lo, hi = max(w, row - radius), min(stats.n_rows - w, row + radius)
    if hi <= lo or row - w < 0 or row + w > stats.n_rows:  # too close to the column's ends to compare
        return row

    def fill(a: int, b: int) -> np.ndarray:
        paper = (b - a) * stats.width - (stats.p_ink[b] - stats.p_ink[a])
        return (stats.p_fill_rgb[b] - stats.p_fill_rgb[a]) / max(paper, 1)

    if np.abs(fill(row - w, row) - fill(row, row + w)).max() > 20:  # a colour boundary
        return row
    ink = np.diff(stats.p_ink[lo : hi + 2]) / stats.width  # ink fraction of rows lo..hi
    blank = ink <= 0.01
    centres, k = [], 0
    while k < len(blank):
        if blank[k]:
            j = k
            while j < len(blank) and blank[j]:
                j += 1
            if j - k >= 2:
                centres.append(lo + (k + j) // 2)
            k = j
        else:
            k += 1
    if not centres:
        return row
    cand = np.array(centres)
    above = scaler.transform(describe_ranges(stats, cand - w, cand))
    below = scaler.transform(describe_ranges(stats, cand, cand + w))
    return int(cand[int(np.argmax(np.linalg.norm(above - below, axis=1)))])


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


def _block_grid(h: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, h // block)
    starts = np.arange(n) * block
    return starts, np.append(starts[1:], h)


def _change_points(X: np.ndarray, min_size: int, penalty: float) -> list[int]:
    """Block indices where the descriptor sequence changes (KernelCPD, linear kernel = L2 cost)."""
    n = len(X)
    if n < 2 * min_size + 1:
        return []
    pen = penalty * X.shape[1] * math.log(n)
    return rpt.KernelCPD(kernel="linear", min_size=min_size).fit(X).predict(pen=pen)[:-1]


def _drop_slivers(stats: RowStats, fscaler: FeatureScaler, rows: list[int], sliver_px: int) -> list[int]:
    """Keep a bed thinner than ``sliver_px`` only when it interrupts one rock (A-b-A).

    Other slivers are transition artefacts where two patterns meet and join the closer side.
    """
    rows = list(rows)
    changed = True
    while changed and len(rows) > 3:
        changed = False
        heights = np.diff(rows)
        for i in np.argsort(heights):
            if heights[i] >= sliver_px:
                break
            if i == 0 or i == len(heights) - 1:
                del rows[1 if i == 0 else i]
                changed = True
                break
            fill = fscaler.transform(
                describe_ranges(stats, np.array(rows[i - 1 : i + 2]), np.array(rows[i : i + 3]))[:, FILL_IDX]
            )
            up, down = np.linalg.norm(fill[1] - fill[0]), np.linalg.norm(fill[1] - fill[2])
            outer = np.linalg.norm(fill[0] - fill[2])
            if outer < min(up, down):  # A-b-A: a genuine thin bed
                continue
            del rows[i if up <= down else i + 1]
            changed = True
            break
    return rows


def _dedupe(boundaries: list[int], h: int, min_gap: int) -> list[int]:
    """[0, sorted boundaries at least ``min_gap`` apart and away from the ends, h]."""
    rows = [0]
    for r in sorted(boundaries):
        if r - rows[-1] >= min_gap and h - r >= min_gap:
            rows.append(r)
    return rows + [h]


def _ignore_thin_lines(fill: np.ndarray, ink_frac: np.ndarray, max_blocks: int) -> np.ndarray:
    """Background colour per block, with ruled lines removed.

    Short runs of all-ink blocks (printed lines) take the colour of the block before; longer runs are dark beds.
    """
    out = fill.copy()
    solid = ink_frac >= 0.9
    i = 0
    while i < len(solid):
        if not solid[i]:
            i += 1
            continue
        j = i
        while j < len(solid) and solid[j]:
            j += 1
        if j - i <= max_blocks:
            src = i - 1 if i > 0 else (j if j < len(out) else None)
            if src is not None:
                out[i:j] = out[src]
        i = j
    return out


def segment_column(rgb: np.ndarray, px_per_ft: float, cfg: SegmentConfig) -> Segmentation:
    """Segment a column crop (H, W, 3); rows in the result are relative to the crop.

    Two passes, merged: colour + pattern on ~1 ft blocks (beds of ``min_thickness_ft`` and more), and fill colour
    only on small blocks (thin beds in colour logs; inert on black-and-white logs).
    """
    stats = row_stats(rgb, cfg.ink_luma)
    h = stats.n_rows

    block = max(cfg.min_block_px, int(round(cfg.block_ft * px_per_ft)))
    starts, ends = _block_grid(h, block)
    desc = describe_ranges(stats, starts, ends)
    scaler = FeatureScaler.from_blocks(desc)
    radius = max(block, int(round(cfg.refine_window_ft * px_per_ft)))
    coarse_min = max(1, int(round(cfg.min_thickness_ft * px_per_ft / block)))
    coarse = [
        _refine(stats, scaler, int(starts[b]), radius, block)
        for b in _change_points(scaler.transform(desc), coarse_min, cfg.penalty)
    ]
    # the pattern pass keeps its own (thicker) minimum, so its noise never produces thin slices
    rows = _absorb_thin(stats, scaler, _dedupe(coarse, h, 2), min_px=int(round(cfg.min_thickness_ft * px_per_ft)))

    fblock = max(cfg.colour_min_block_px, int(round(cfg.colour_block_ft * px_per_ft)))
    # a bed must also span a few pixel rows: below that a low-resolution log cannot resolve it
    thin_px = max(cfg.colour_min_px, int(round(cfg.colour_min_thickness_ft * px_per_ft)))
    fstarts, fends = _block_grid(h, fblock)
    full = describe_ranges(stats, fstarts, fends)
    fdesc = _ignore_thin_lines(full[:, FILL_IDX], full[:, 6], max_blocks=max(1, thin_px // fblock))
    fscaler = FeatureScaler.from_blocks(fdesc, floor=DEFAULT_FLOOR[FILL_IDX])
    fine_min = max(1, int(round(thin_px / fblock)))
    sliver_px = max(thin_px, int(round(cfg.colour_sliver_ft * px_per_ft)))
    w_fine = max(2, sliver_px // 2)  # refinement windows wider than a pattern's line spacing
    fine = [
        _refine(stats, fscaler, int(fstarts[b]), 2 * fblock, w_fine, FILL_IDX)
        for b in _change_points(fscaler.transform(fdesc), fine_min, cfg.colour_penalty)
    ]
    rows = _dedupe(rows[1:-1] + fine, h, max(2, thin_px // 2))
    rows = _absorb_thin(stats, scaler, rows, min_px=thin_px)
    rows = _drop_slivers(stats, fscaler, rows, sliver_px)

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
