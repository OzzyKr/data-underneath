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
    """Prefix sums over the rows of an (H, W, 3) crop.

    Any row range [a, b) is summarised in O(1) per statistic (``P[b] - P[a]``), so descriptors for thousands of
    candidate ranges (block grids, boundary refinement) cost nothing extra on 300k-row columns.
    """

    width: int
    n_rows: int
    p_ink: np.ndarray  # (H+1,) ink pixel count
    p_ink_rgb: np.ndarray  # (H+1, 3) sum of ink pixel colours
    p_fill_rgb: np.ndarray  # (H+1, 3) sum of non-ink pixel colours
    p_h: np.ndarray  # (H+1,) horizontal ink adjacencies
    p_v: np.ndarray  # (H+1,)
    p_d: np.ndarray  # (H+1,)
    p_rf: np.ndarray  # (H+1,) per-row ink fraction
    p_rf2: np.ndarray  # (H+1,) its square
    p_col: np.ndarray  # (H+1, W) int32 ink count per column

    def range_sum(self, p: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return p[ends] - p[starts]


def _prefix(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1,) + a.shape[1:], dtype=np.float64)
    np.cumsum(a, axis=0, out=out[1:])
    return out


def row_stats(rgb: np.ndarray, ink_luma: int, chunk: int = 8192) -> RowStats:
    """Per-row statistics as prefix sums, computed in row chunks so temporaries stay small."""
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
    p_col = np.zeros((h_total + 1, width), np.int32)
    np.cumsum(ink, axis=0, dtype=np.int32, out=p_col[1:])
    rf = out_ink / width
    return RowStats(
        width=width,
        n_rows=h_total,
        p_ink=_prefix(out_ink),
        p_ink_rgb=_prefix(out_ink_rgb),
        p_fill_rgb=_prefix(out_fill_rgb),
        p_h=_prefix(out_h),
        p_v=_prefix(out_v),
        p_d=_prefix(out_d),
        p_rf=_prefix(rf),
        p_rf2=_prefix(rf**2),
        p_col=p_col,
    )


def _lab(rgb_mean: np.ndarray) -> np.ndarray:
    """Convert (N, 3) mean RGB in 0..255 to Lab."""
    return rgb2lab(np.clip(rgb_mean / 255.0, 0, 1)[None])[0]


def describe_ranges(stats: RowStats, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Descriptors (N, 12) for row ranges [starts[i], ends[i]) of a crop."""
    starts = np.asarray(starts, int)
    ends = np.asarray(ends, int)
    n_px = (ends - starts) * stats.width

    def seg(p: np.ndarray) -> np.ndarray:
        return stats.range_sum(p, starts, ends)

    ink = seg(stats.p_ink)
    fill_n = n_px - ink
    ink_rgb = seg(stats.p_ink_rgb)
    fill_rgb = seg(stats.p_fill_rgb)
    safe_ink = np.maximum(ink, 1)[:, None]
    safe_fill = np.maximum(fill_n, 1)[:, None]
    fill_mean = np.where(fill_n[:, None] > 0, fill_rgb / safe_fill, ink_rgb / safe_ink)
    ink_mean = np.where(ink[:, None] > 0, ink_rgb / safe_ink, fill_mean)

    nrows = np.maximum(ends - starts, 1)
    rf, rf2 = seg(stats.p_rf), seg(stats.p_rf2)
    row_spread = np.sqrt(np.maximum(rf2 / nrows - (rf / nrows) ** 2, 0))
    col_frac = (stats.p_col[ends] - stats.p_col[starts]) / nrows[:, None]
    col_spread = col_frac.std(axis=1)

    return np.column_stack(
        [
            _lab(fill_mean),
            _lab(ink_mean),
            ink / np.maximum(n_px, 1),
            seg(stats.p_h) / safe_ink[:, 0],
            seg(stats.p_v) / safe_ink[:, 0],
            seg(stats.p_d) / safe_ink[:, 0],
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
