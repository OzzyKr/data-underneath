"""Pattern descriptors shared by segmentation, clustering and legend matching.

A lithology fill is a background colour plus a repeated ink pattern (dots, dashes, bricks, hatching). The
descriptor captures both with cheap, explainable statistics:

  fill colour (Lab)   mean colour of non-ink pixels
  ink colour (Lab)    mean colour of ink pixels
  ink fraction        how much of the area is ink
  h / v / d adjacency share of ink pixels whose right / lower / diagonal neighbour is also ink
                      (horizontal dashes -> high h, vertical dashes -> high v, dots -> low everything)
  row / col spread    variation of ink between rows / columns (ruled patterns vs uniform stipple)

Everything is computed per pixel row first and then summed over blocks, so a 300,000-row column costs a few
vectorised passes rather than a Python loop per block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

FEATURE_NAMES: tuple[str, ...] = (
    "fill_L",
    "fill_a",
    "fill_b",
    "ink_L",
    "ink_a",
    "ink_b",
    "ink_frac",
    "h_adj",
    "v_adj",
    "d_adj",
    "row_spread",
    "col_spread",
)
COLOUR_IDX = np.array([0, 1, 2, 3, 4, 5])
TEXTURE_IDX = np.array([6, 7, 8, 9, 10, 11])


@dataclass
class RowStats:
    """Per-row sums for an (H, W, 3) crop; block sums are differences of cumulative sums."""

    width: int
    ink: np.ndarray  # (H,) ink pixel count
    ink_rgb: np.ndarray  # (H, 3) sum of ink pixel colours
    fill_rgb: np.ndarray  # (H, 3) sum of non-ink pixel colours
    h_adj: np.ndarray  # (H,)
    v_adj: np.ndarray  # (H,)
    d_adj: np.ndarray  # (H,)
    ink_mask: np.ndarray  # (H, W) bool, kept for column statistics

    @property
    def n_rows(self) -> int:
        return len(self.ink)


def row_stats(rgb: np.ndarray, ink_luma: int, chunk: int = 8192) -> RowStats:
    """Per-row sums, computed in row chunks so temporaries stay small on 300k-row columns."""
    h_total, width = rgb.shape[:2]
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    ink = luma < ink_luma
    del luma
    out_ink = np.empty(h_total)
    out_ink_rgb = np.empty((h_total, 3))
    out_fill_rgb = np.empty((h_total, 3))
    out_h = np.empty(h_total)
    out_v = np.zeros(h_total)
    out_d = np.zeros(h_total)
    for a in range(0, h_total, chunk):
        b = min(h_total, a + chunk)
        m = ink[a:b]
        c = rgb[a:b].astype(np.float32)
        out_ink[a:b] = m.sum(1)
        out_ink_rgb[a:b] = (c * m[..., None]).sum(1)
        out_fill_rgb[a:b] = c.sum(1) - out_ink_rgb[a:b]
        out_h[a:b] = (m[:, :-1] & m[:, 1:]).sum(1)
        b2 = min(h_total, b + 1)  # one extra row so vertical neighbours cross the chunk edge
        m2 = ink[a:b2]
        n = m2.shape[0] - 1
        if n > 0:
            out_v[a : a + n] = (m2[:-1] & m2[1:]).sum(1)
            out_d[a : a + n] = ((m2[:-1, :-1] & m2[1:, 1:]) | (m2[:-1, 1:] & m2[1:, :-1])).sum(1)
    return RowStats(
        width=width,
        ink=out_ink,
        ink_rgb=out_ink_rgb,
        fill_rgb=out_fill_rgb,
        h_adj=out_h,
        v_adj=out_v,
        d_adj=out_d,
        ink_mask=ink,
    )


def _lab(rgb_mean: np.ndarray) -> np.ndarray:
    """Convert (N, 3) mean RGB in 0..255 to Lab."""
    return rgb2lab(np.clip(rgb_mean / 255.0, 0, 1)[None])[0]


def describe_ranges(stats: RowStats, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Descriptors (N, 12) for row ranges [starts[i], ends[i]) of a crop."""
    starts = np.asarray(starts, int)
    ends = np.asarray(ends, int)
    n_px = (ends - starts) * stats.width

    def seg(a: np.ndarray) -> np.ndarray:
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
        return c[ends] - c[starts]

    ink = seg(stats.ink)
    fill_n = n_px - ink
    ink_rgb = seg(stats.ink_rgb)
    fill_rgb = seg(stats.fill_rgb)
    safe_ink = np.maximum(ink, 1)[:, None]
    safe_fill = np.maximum(fill_n, 1)[:, None]
    fill_mean = np.where(fill_n[:, None] > 0, fill_rgb / safe_fill, ink_rgb / safe_ink)
    ink_mean = np.where(ink[:, None] > 0, ink_rgb / safe_ink, fill_mean)

    row_frac = stats.ink / stats.width
    rf = seg(row_frac)
    rf2 = seg(row_frac**2)
    nrows = np.maximum(ends - starts, 1)
    row_spread = np.sqrt(np.maximum(rf2 / nrows - (rf / nrows) ** 2, 0))
    # column spread: per-range ink count per column
    colsum = np.zeros((stats.n_rows + 1, stats.width), np.int32)
    np.cumsum(stats.ink_mask, axis=0, dtype=np.int32, out=colsum[1:])
    col_frac = (colsum[ends] - colsum[starts]) / nrows[:, None]
    col_spread = col_frac.std(axis=1)

    return np.column_stack(
        [
            _lab(fill_mean),
            _lab(ink_mean),
            ink / np.maximum(n_px, 1),
            seg(stats.h_adj) / safe_ink[:, 0],
            seg(stats.v_adj) / safe_ink[:, 0],
            seg(stats.d_adj) / safe_ink[:, 0],
            row_spread,
            col_spread,
        ]
    )


def describe_patch(rgb: np.ndarray, ink_luma: int) -> np.ndarray:
    """Descriptor (12,) of a whole patch, e.g. a legend swatch."""
    st = row_stats(rgb, ink_luma)
    return describe_ranges(st, np.array([0]), np.array([st.n_rows]))[0]


MATCH_NAMES: tuple[str, ...] = (
    "fill_L",
    "fill_a",
    "fill_b",
    "ink_L",
    "ink_a",
    "ink_b",
    "skel_density",
    "skel_h",
    "skel_v",
    "skel_d",
)
# Fixed scales for column-vs-legend matching: ~10 Lab units is a clearly different colour; texture terms are
# fractions. Fixed (not learned) so the matcher has nothing to overfit.
MATCH_SCALE = np.array([10, 10, 10, 15, 15, 15, 0.02, 0.15, 0.15, 0.15])


def match_descriptor(rgb: np.ndarray, ink_luma: int) -> np.ndarray:
    """Descriptor for comparing a column patch with a legend swatch drawn at a different stroke width.

    Texture is measured on the skeleton (strokes thinned to 1 px), so thick and thin renderings of the same
    pattern agree: horizontal dashes give a high skel_h, vertical bars a high skel_v, dots neither.
    """
    from skimage.morphology import skeletonize

    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    ink = luma < ink_luma
    rgbf = rgb.reshape(-1, 3).astype(np.float64)
    flat = ink.ravel()
    fill = rgbf[~flat].mean(0) if (~flat).any() else rgbf[flat].mean(0)
    inkc = rgbf[flat].mean(0) if flat.any() else fill
    sk = skeletonize(ink) if ink.any() else ink
    n = max(int(sk.sum()), 1)
    return np.concatenate(
        [
            _lab(np.array([fill]))[0],
            _lab(np.array([inkc]))[0],
            [
                sk.mean(),
                (sk[:, :-1] & sk[:, 1:]).sum() / n,
                (sk[:-1] & sk[1:]).sum() / n,
                ((sk[:-1, :-1] & sk[1:, 1:]) | (sk[:-1, 1:] & sk[1:, :-1])).sum() / n,
            ],
        ]
    )


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature scale from the column's own block-to-block noise, so distances are in noise units."""

    scale: np.ndarray

    @classmethod
    def from_blocks(cls, desc: np.ndarray, floor: np.ndarray | None = None) -> FeatureScaler:
        diffs = np.abs(np.diff(desc, axis=0))
        mad = np.median(diffs, axis=0) / 0.6745 / np.sqrt(2) if len(desc) > 2 else np.ones(desc.shape[1])
        floor = floor if floor is not None else DEFAULT_FLOOR
        return cls(np.maximum(mad, floor))

    def transform(self, desc: np.ndarray) -> np.ndarray:
        return desc / self.scale


# Smallest meaningful difference per feature (Lab units ~ just-noticeable colour difference; fractions).
DEFAULT_FLOOR = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 0.01, 0.03, 0.03, 0.03, 0.01, 0.01])
