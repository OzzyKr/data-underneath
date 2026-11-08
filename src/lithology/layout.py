"""Bonus: find the three boxes (depth column, interpreted lithology column, lithology legend) without the
organisers' annotation file.

Classical CV plus the pipeline's own Tesseract OCR. No training and no stored templates, so the detector cannot
memorise any vendor's page layout. Propose, score, select:

- **body**: the rows spanned by the long vertical track rules. The header ends at the lowest full-width rule just
  below where those rules begin.
- **depth column**: numeric OCR tokens (Tesseract ``--psm 11``, digits only, same config as ``scale.py``) in a few
  bands across the full page width, in whichever orientation the labels are printed. Tokens are clustered by x.
  A cluster whose values fit one rising line, or that is regularly spaced (abbreviated labels such as 20/40/60),
  is a depth track. It is snapped to the nearest vertical rules.
- **legend**: candidates are the header-table cells (closed rule rectangles), the bands between full-width rules,
  and the box around the rock-name words. They are scored by rock-word share and purity, a legend title, and
  non-rock text.
- **lithology column**: candidates are intervals between vertical edges (rules and the edges of solid bands).
  They are scored by row-uniform fill, *one* pattern per row (a percent-lithology track has several), coverage
  of the legend's colours, ink spread across the width, and a "Lith…" header above.

Every box comes with a confidence. Diagnostics list the runner-up candidates.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image

from lithology.config import LayoutConfig
from lithology.io import crop_rgb, image_dpi
from lithology.legend import MODIFIER_WORDS, ROCK_WORDS, _ocr_words, correct_word
from lithology.models import Box, DocumentBoxes
from lithology.scale import _TESS_CONFIG, hypotheses

log = logging.getLogger(__name__)

# digits only, sparse text; no second inverted pass (low-confidence digits on fills would trigger it on every band)
_DIGITS_CFG = _TESS_CONFIG + " -c tessedit_do_invert=0"
_TITLE = re.compile(r"litholog|legend|symbol|rock\s*types?|lithologies", re.I)
_BAD_HEADER = ("%", "desc", "cutt", "percent", "comment", "remark", "show")
_NEG_HEADER = ("cutt", "%", "percent", "desc")
_META = re.compile(
    r"operator|company|\bapi\b|scale|date|\bwell\b|county|field|elevation|spud|contractor|\brig\b|location|"
    r"state|country|permit|licen[cs]e|hole size|casing|mud type|logged by|geologist|prepared",
    re.I,
)


@dataclass
class LayoutResult:
    boxes: DocumentBoxes
    confidence: dict[str, float]
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------
# Page access (rows are read lazily from the PIL image; the pipeline decodes the page anyway)


def _gray(a: np.ndarray) -> np.ndarray:
    """Luminance (for OCR)."""
    if a.ndim == 2:
        return a
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)


def _ink(a: np.ndarray) -> np.ndarray:
    """Darkest-possible view for rules: a pixel is dark only if ALL channels are dark, so coloured fills
    (yellow sand, green shale, cyan limestone) never look like black rule lines."""
    return a if a.ndim == 2 else a.max(axis=2)


def _noise_sigma(img: Image.Image, n: int = 40) -> float:
    """Pixel noise of the page (grey levels): robust spread of horizontal differences between paper-white pixels.
    Clean renders and bilevel scans give 0."""
    W, H = img.size
    ys = np.linspace(0, H - 1, min(n, H)).astype(int)
    g = np.stack([np.asarray(img.crop((0, int(y), W, int(y) + 1)).convert("L"))[0] for y in ys]).astype(np.int16)
    paper = (g[:, 1:] > 190) & (g[:, :-1] > 190)
    if paper.sum() < 1000:
        return 0.0
    v = (g[:, 1:] - g[:, :-1])[paper]
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(2))


def _median_x(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Median over k pixels along x of independent sampled rows (no mixing between rows)."""
    r = k // 2
    pad = np.pad(a, [(0, 0), (r, r)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    return np.median(np.stack([pad[:, i : i + a.shape[1]] for i in range(k)]), axis=0).astype(a.dtype)


def _rows(img: Image.Image, ys: np.ndarray, x0: int = 0, x1: int | None = None) -> np.ndarray:
    W = img.size[0]
    x1 = W if x1 is None else x1
    return np.stack([np.asarray(img.crop((x0, int(y), x1, int(y) + 1)).convert("RGB"))[0] for y in ys])


def _mm(dpi: float, mm: float) -> float:
    return mm / 25.4 * dpi


# ---------------------------------------------------------------------------------------------------------------
# Body


def _thin_runs(xs: np.ndarray, max_w: int) -> np.ndarray:
    """Keep the columns of contiguous runs no wider than max_w px."""
    if not len(xs):
        return xs
    cuts = np.where(np.diff(xs) > 1)[0] + 1
    return np.concatenate([r for r in np.split(xs, cuts) if len(r) <= max_w] or [np.zeros(0, int)])


def body_range(img: Image.Image, dpi: float, cfg: LayoutConfig) -> tuple[int, int, np.ndarray]:
    """(top, bottom, x of the long vertical rules). Top is snapped to the header's bottom rule."""
    W, H = img.size
    ys = np.linspace(0, H - 1, min(cfg.body_samples, H)).astype(int)
    dark = _ink(_rows(img, ys)) < 150
    shift = estimate_skew(dark, ys, W)
    dark = deskew_rows(dark, ys, shift)
    n = len(ys)
    inner = (np.arange(W) > 0.015 * W) & (np.arange(W) < 0.985 * W)
    # 1. columns dark over a large part of the page are rule candidates; the rows where most of them are dark are
    #    the "structured" rows (no fixed band: logs may end early or start late)
    cand = np.where((dark.mean(axis=0) > 0.3) & inner)[0]
    if len(cand) < 2:
        return int(0.1 * H), H, cand
    structured = dark[:, cand].mean(axis=1) >= 0.6
    if structured.sum() < 10:
        structured = np.ones(n, bool)
    # 2. per-x rule fraction over the structured rows, threshold relaxed until at least two rules remain
    frac = dark[structured].mean(axis=0)
    xs = np.zeros(0, int)
    for thr in (0.9, 0.75, 0.6):
        xs = np.where((frac >= thr) & inner)[0]
        xs = _thin_runs(xs, max(3, round(_mm(dpi, 1.0))))  # rules are thin: wide runs are dark fill, not rules
        if len(xs) >= 2:
            break
    if len(xs) < 2:
        return int(0.1 * H), H, xs
    # 3. rows where the rules persist; bridge gaps < 0.15 in; body = first long run .. last long run
    on = dark[:, xs].mean(axis=1) >= 0.7
    step = max(1.0, (H - 1) / max(1, n - 1))
    gap = max(1, int(cfg.body_gap_in * dpi / step))
    runs = []
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=False):
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    long_len = max(2, int(1.0 * dpi / step))
    long_runs = [r for r in runs if r[1] - r[0] >= long_len] or (sorted(runs, key=lambda r: r[0] - r[1])[:1])
    if not long_runs:
        return int(0.1 * H), H, xs
    a, b = long_runs[0][0], long_runs[-1][1] - 1
    top, bot = int(ys[a]), int(min(H, ys[b] + 1))
    # header bottom rule: lowest near-full-width dark row in the first inches of the structure. (A mirrored snap at
    # the bottom, onto a repeated footer header, was tried and removed: lone depth grid lines fooled it.)
    xa, xb = int(xs.min()), int(xs.max()) + 1
    top = max(0, top - int(step))  # the run starts at the first sample past the rule; look from the sample before
    win_h = int(min(bot - top, cfg.header_search_in * dpi))
    if win_h > 10:
        g = _ink(crop_rgb(img, Box(xa, top, xb, top + win_h))) < 150
        cand = np.where(g.mean(axis=1) > 0.85)[0]
        lines = _row_lines(cand, top)
        # body grid lines are full-width too: on a regular depth grid, follow it up to its first line (the header rule)
        P = _grid_period(img, xa, xb, top + int(step) + int(0.2 * dpi), min(bot, top + int(8 * dpi)), dpi)
        start = lines[-1] if lines else None
        if P and start is None:
            ls = _full_rows(img, xa, xb, top, int(top + P + 4), 0.75)
            start = ls[0] if ls else None
        if start is not None:
            s0, e0 = start
            moved = False
            while P:
                up = _full_rows(img, xa, xb, max(0, int(s0 - P) - 4), max(1, int(s0 - P) + 5), 0.85)
                if not up:
                    break
                (s0, e0), moved = up[0], True
            if lines or moved:
                top = e0 + 1
    return top, bot, xs


def _row_lines(rows: np.ndarray, y0: int) -> list[tuple[int, int]]:
    """Group row indices (offset by y0) into lines: runs with gaps of at most 2 px."""
    lines: list[list[int]] = []
    for y in rows:
        if lines and y - lines[-1][1] <= 2:
            lines[-1][1] = int(y)
        else:
            lines.append([int(y), int(y)])
    return [(y0 + a, y0 + b) for a, b in lines]


def _full_rows(img: Image.Image, xa: int, xb: int, y0: int, y1: int, thr: float) -> list[tuple[int, int]]:
    """Horizontal lines in rows y0..y1 whose ink covers more than thr of the span xa..xb."""
    if y1 - y0 < 1:
        return []
    g = _ink(crop_rgb(img, Box(xa, y0, xb, y1))) < 150
    return _row_lines(np.where(g.mean(axis=1) > thr)[0], y0)


def _grid_period(img: Image.Image, xa: int, xb: int, y0: int, y1: int, dpi: float) -> float | None:
    """Spacing of a regular depth grid (full-width lines at a fixed step) in rows y0..y1, or None."""
    ls = _full_rows(img, xa, xb, y0, y1, 0.75)
    if len(ls) < 3:
        return None
    d = np.diff([a for a, _ in ls]).astype(float)
    med = float(np.median(d))
    if med < 0.1 * dpi or (np.abs(d - med) <= max(2, 0.02 * med)).mean() < 0.6:
        return None
    return med


def estimate_skew(dark: np.ndarray, ys: np.ndarray, W: int, n_bands: int = 20) -> np.ndarray:
    """Per-row horizontal drift of the vertical rules (scan skew, curved drift or stitched scans), in px.
    Each band's dark-column profile is cross-correlated with a reference band from the log body; only bands that
    correlate well (normalised peak >= 0.6, so header bands are ignored) are used, and the shift is interpolated
    between them."""
    n = len(ys)
    zero = np.zeros(n)
    if n < 4 * n_bands:
        return zero
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    prof = [dark[a:b].mean(axis=0).astype(np.float32) for a, b in zip(edges[:-1], edges[1:], strict=False)]
    # reference: the band with the most columns dark on (nearly) every row, i.e. inside the log body. A fixed
    # position can fall in the header or below a log that ends early, where nothing correlates with it.
    ref_i = int(np.argmax([(p >= 0.9).sum() for p in prof]))
    ref = prof[ref_i] - prof[ref_i].mean()
    if np.linalg.norm(ref) < 1e-3:
        return zero
    maxs = max(4, int(0.03 * W))
    yc, sh = [], []
    for (a, b), p in zip(zip(edges[:-1], edges[1:], strict=False), prof, strict=False):
        q = p - p.mean()
        nq = float(np.linalg.norm(q))
        if nq < 1e-3:
            continue
        cc = np.array([float(np.dot(np.roll(q, -k), ref)) for k in range(-maxs, maxs + 1)])
        k = int(np.argmax(cc))
        if cc[k] / (nq * float(np.linalg.norm(ref))) >= 0.6:
            yc.append(float(ys[(a + b) // 2]))
            sh.append(float(k - maxs))
    if len(sh) < 3 or max(sh) - min(sh) < 3:  # no measurable drift
        return zero
    return np.interp(ys, yc, sh)


def _straight(rows_rgb: np.ndarray) -> np.ndarray:
    """Sampled RGB rows (top to bottom) with the scan's horizontal drift removed, so the track rules line up."""
    n, W = rows_rgb.shape[:2]
    shift = estimate_skew(_ink(rows_rgb) < 150, np.arange(n), W)
    if not np.any(np.abs(shift) >= 1):
        return rows_rgb
    return np.stack([np.roll(r, -int(round(v)), axis=0) for r, v in zip(rows_rgb, shift, strict=False)])


def deskew_rows(dark: np.ndarray, ys: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Shift each sampled row back by its estimated drift."""
    if not np.any(np.abs(shift) >= 1):
        return dark
    out = np.empty_like(dark)
    for i in range(len(ys)):
        out[i] = np.roll(dark[i], -int(round(shift[i])))
    return out


def vertical_edges(rows_rgb: np.ndarray, dpi: float) -> tuple[list[int], np.ndarray, list[tuple[int, int]]]:
    """Edges = centres of thin rules + both sides of thick dark bands. Also returns the per-x dark fraction and the
    full x-extent of every thin rule (so a candidate starting at a rule's centre is not mistaken for containing it:
    at 400 dpi a rule is ~8 px wide)."""
    dark = (_ink(rows_rgb) < 150).mean(axis=0)
    W = len(dark)
    thin = max(3, int(round(_mm(dpi, 0.6))))
    edges = {0, W}
    rules: list[tuple[int, int]] = []
    m = dark > 0.55
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=False):
        if e - s <= thin:
            edges.add(int((s + e) // 2))
            if dark[s:e].max() > 0.9:
                rules.append((int(s), int(e)))
        else:
            edges.add(int(s))
            edges.add(int(e))
    for x in fill_edges(rows_rgb, dpi):  # a coloured column often ends where its fill meets white paper (no rule)
        edges.add(int(x))
    return sorted(edges), dark, rules


def fill_edges(rows_rgb: np.ndarray, dpi: float, min_step: float = 0.3, min_col: float = 40.0) -> list[int]:
    """x positions where the per-column fill fraction or mean fill colour changes sharply (non-max suppressed)."""
    s = rows_rgb.astype(np.int16)
    mn, mx = s.min(axis=2), s.max(axis=2)
    fill = ((mn < 215) | ((mx - mn) > 35)) & (mx >= 90)  # coloured / grey fill, not black ink
    f = fill.mean(axis=0).astype(np.float32)
    cnt = fill.sum(axis=0)[:, None].astype(np.float32)
    col = np.where(cnt > 0, (s * fill[..., None]).sum(axis=0) / np.maximum(cnt, 1), 255.0)  # mean fill RGB per x
    d = max(2, int(round(_mm(dpi, 0.5))))  # non-max suppression radius
    w = max(4, int(round(_mm(dpi, 1.5))))  # compare 1.5 mm means on each side: thin grid lines average out
    W = len(f)
    if len(f) <= 2 * w + 1:
        return []
    cf = np.concatenate([[0.0], np.cumsum(f, dtype=np.float64)])
    cc = np.concatenate([np.zeros((1, 3)), np.cumsum(col, axis=0, dtype=np.float64)])
    xs = np.arange(w, W - w)
    left_f, right_f = (cf[xs] - cf[xs - w]) / w, (cf[xs + w] - cf[xs]) / w
    left_c, right_c = (cc[xs] - cc[xs - w]) / w, (cc[xs + w] - cc[xs]) / w
    step = np.zeros(W, np.float32)
    step[xs] = np.abs(right_f - left_f)
    cstep = np.zeros(W, np.float32)
    cstep[xs] = np.linalg.norm(right_c - left_c, axis=1)
    strength = np.maximum(step / min_step, cstep / min_col)
    out: list[int] = []
    for x in np.argsort(-strength):
        if strength[x] < 1.0:
            break
        if all(abs(int(x) - o) > d for o in out):
            out.append(int(x))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------
# Depth column


def _ocr_numbers(arr: np.ndarray, rot: int) -> list[tuple[float, float, float, float]]:
    """(x, y, value, half-width in x) of digit tokens in ``arr`` read at rotation ``rot`` (0, 90 = CCW, 270 = CW)."""
    pil = Image.fromarray(arr)
    h, w = arr.shape[:2]
    if rot == 90:
        pil = pil.transpose(Image.Transpose.ROTATE_90)
    elif rot == 270:
        pil = pil.transpose(Image.Transpose.ROTATE_270)
    d = pytesseract.image_to_data(pil, config=_DIGITS_CFG, output_type=pytesseract.Output.DICT)
    out = []
    for t, c, x, y, bw, bh in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"], strict=False):
        if float(c) < 0:  # Tesseract reports -1 for non-words; 0 is common for good digits on wide bands
            continue
        hyp = [v for v, primary in hypotheses(t, 45000) if primary]
        if not hyp:
            continue
        u, v = x + bw / 2, y + bh / 2
        if rot == 90:  # rotated (u, v) -> original (x = w-1-v, y = u)
            ox, oy, hw = w - 1 - v, u, bh / 2
        elif rot == 270:  # rotated (u, v) -> original (x = v, y = h-1-u)
            ox, oy, hw = v, h - 1 - u, bh / 2
        else:
            ox, oy, hw = u, v, bw / 2
        out.append((float(ox), float(oy), float(hyp[0]), float(hw)))
    return out


def _rotation_score(found: list[tuple], dpi: float, cfg: LayoutConfig) -> int:
    """How well a rotation's tokens line up as depth labels: best line-fit inlier count over x-clusters."""
    full = sorted((x, y, v) for x, y, v, _ in found if v >= 100)
    if len(full) < 2:
        return len(full)
    tol = cfg.depth_cluster_in * dpi
    best, cur = 0, [full[0]]
    for t in full[1:] + [(float("inf"), 0, 0)]:
        if t[0] - cur[-1][0] <= tol:
            cur.append(t)
            continue
        if len(cur) >= 2:
            _, _, inl = _line_fit(
                [(y, v) for _, y, v in cur], tol=max(5.0, 0.02 * float(np.median([v for *_, v in cur])))
            )
            best = max(best, inl)
        cur = [t]
    return best


def number_tokens(
    img: Image.Image,
    top: int,
    bot: int,
    dpi: float,
    cfg: LayoutConfig,
    fracs: np.ndarray | None = None,
    band_in: float | None = None,
    rots: list[int] | None = None,
) -> tuple[list[tuple], list[int]]:
    """Digit tokens (x, y, value, half-width) in body bands, and the label orientation(s) used.
    Without ``rots`` the first band is read at 0/90/270 and the orientation whose tokens best fit a rising line is
    kept (count fallback); the remaining bands are read concurrently (Tesseract runs as a subprocess)."""
    from concurrent.futures import ThreadPoolExecutor

    W = img.size[0]
    bh = int((band_in or cfg.depth_band_in) * dpi)
    sc = 0.5 if dpi >= 300 else 1.0
    fracs = np.linspace(0.12, 0.88, cfg.depth_bands) if fracs is None else fracs
    bands = []
    for f in fracs:
        yc = int(top + f * (bot - top))
        a, b = max(top, yc - bh // 2), min(bot, yc + bh // 2)
        if b - a >= 20:
            bands.append((a, _gray(crop_rgb(img, Box(0, a, W, b), scale=sc))))
    if not bands:
        return [], list(rots or [0])

    def read(a: int, band: np.ndarray, r: int) -> list[tuple]:
        return [(x / sc, a + y / sc, v, hw / sc) for x, y, v, hw in _ocr_numbers(band, r)]

    toks: list[tuple] = []
    start = 0
    if not rots:
        # judge the orientation on the first two bands together; trust it only when >= 3 labels share one line
        # (any two noise tokens "fit" a line); otherwise keep every orientation, so accuracy is never traded away
        probe = bands[:2]
        jobs0 = [(a, band, r) for a, band in probe for r in (0, 90, 270)]
        with ThreadPoolExecutor(3) as ex:
            outs = list(ex.map(lambda j: read(*j), jobs0))
        by_rot: dict[int, list] = {0: [], 90: [], 270: []}
        for (_, _, r), out in zip(jobs0, outs, strict=False):
            by_rot[r] += out
            toks += out
        score = {r: _rotation_score(t, dpi, cfg) for r, t in by_rot.items()}
        best = max(score.values())
        rots = [r for r in (0, 90, 270) if score[r] >= 0.7 * best] if best >= 3 else [0, 90, 270]
        start = len(probe)
    jobs = [(a, band, r) for a, band in bands[start:] for r in rots]
    with ThreadPoolExecutor(3) as ex:
        for out in ex.map(lambda j: read(*j), jobs):  # map keeps the job order: identical token order
            toks += out
    return toks, list(rots)


def _line_fit(pts: list[tuple[float, float]], tol: float, iters: int = 300) -> tuple[float | None, float, int]:
    if len(pts) < 2:
        return None, 0.0, 0
    P = np.array(pts, float)
    y, d = P[:, 0], P[:, 1]
    rng = np.random.default_rng(0)
    best = (0, None, 0.0)
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        if abs(y[j] - y[i]) < 1:
            continue
        a = (d[j] - d[i]) / (y[j] - y[i])
        if a <= 0:
            continue
        b = d[i] - a * y[i]
        n = int((np.abs(a * y + b - d) < tol).sum())
        if n > best[0]:
            best = (n, a, b)
    return best[1], best[2], best[0]


def depth_candidates(
    toks: list[tuple], edges: list[int], dark: np.ndarray, dpi: float, cfg: LayoutConfig, top: int, bot: int, W: int
) -> list[dict]:
    if not toks:
        return []
    T = np.array(sorted(toks))
    tol = cfg.depth_cluster_in * dpi
    clusters, cur = [], [T[0]]
    for t in T[1:]:
        if t[0] - cur[-1][0] <= tol:
            cur.append(t)
        else:
            clusters.append(np.array(cur))
            cur = [t]
    clusters.append(np.array(cur))
    reach, min_w = cfg.depth_snap_in * dpi, _mm(dpi, cfg.min_track_mm)
    out = []
    for c in clusters:
        if len(c) < 3:
            continue
        full = [(y, v) for _, y, v, _ in c if v >= 100]
        a, b, inl = _line_fit(full, tol=max(5.0, 0.02 * np.median([v for _, v in full]) if full else 5.0))
        regular = False
        if a is None or inl < 3 or not (0.005 <= a <= 2.0):
            ys = np.sort(c[:, 1])
            keep = [ys[0]]
            for yv in ys[1:]:
                if yv - keep[-1] > 0.05 * dpi:
                    keep.append(yv)
            dd = np.diff(np.array(keep))
            dd = dd[(dd > 0.05 * dpi) & (dd < 2.2 * dpi)]
            if len(dd) >= 3 and np.std(dd) / (np.median(dd) + 1e-9) < 0.6:
                regular, inl = True, len(dd) + 1
            else:
                continue
        xl, xr = float((c[:, 0] - c[:, 3]).min()), float((c[:, 0] + c[:, 3]).max())  # label extents, not centres
        # tick marks on the scale's rule are read into the labels: accept a rule up to ~0.03 in inside the left
        # label extent (but never past the label centres)
        tuck = int(0.03 * dpi)
        cl = float(np.percentile(c[:, 0], 10))
        lo = int(max(0, xl - reach))
        left = np.where(dark[lo : int(min(xl + tuck, cl))] > 0.6)[0]
        x0 = lo + int(left.max()) + 1 if len(left) else int(max(0, xl - 0.1 * dpi))
        hi = int(min(W, xr + reach))
        start = int(xr) + 1  # (right side: no tuck; label ends there are the digits themselves)
        right = np.where(dark[start:hi] > 0.6)[0]
        x1 = start + int(right.min()) if len(right) else int(min(W, xr + 0.1 * dpi))
        if x1 - x0 < min_w:
            pad = int((min_w - (x1 - x0)) / 2) + 1
            x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
        score = (0.5 if regular else 1.0) * inl / (1.0 + len(c) - inl)
        out.append(
            {
                "box": (x0, top, x1, bot),
                "labels": (int(xl), int(xr) + 1),
                "inliers": inl,
                "n": len(c),
                "regular_only": regular,
                "score": score,
            }
        )
    return sorted(out, key=lambda d: (d["regular_only"], -d["score"]))  # a line fit beats regular spacing


# ---------------------------------------------------------------------------------------------------------------
# Legend


def _phrases(words: list, gap_h: float = 1.5) -> list[list]:
    """Group words (x0, y0, x1, y1, text) into horizontal phrases: same text line, small x gaps."""
    ws = sorted(words, key=lambda w: (round((w[1] + w[3]) / 2), w[0]))
    lines: list[list] = []
    for w in sorted(ws, key=lambda w: (w[1] + w[3]) / 2):
        h = max(1.0, w[3] - w[1])
        cy = (w[1] + w[3]) / 2
        for ln in lines:
            last = ln[-1]
            if abs((last[1] + last[3]) / 2 - cy) < 0.6 * h and 0 <= w[0] - last[2] < gap_h * h:
                ln.append(w)
                break
        else:
            lines.append([w])
    return lines


_POS_HEADER = re.compile(r"lith|interp|\bint\.?\s*lit")
_STRONG_HEADER = re.compile(r"interp|\bint\.?\s*lit")  # "Interp Lith", "Int. Lith": the interpreted column by name


def _rule_top(dk: np.ndarray, x: int, gap: int) -> int:
    """Topmost row of the vertical rule at column x, followed upward from the bottom row across gaps < gap rows."""
    col = dk[:, max(0, x - 1) : x + 2].any(axis=1)
    y, last = len(col) - 1, len(col) - 1
    while y >= 0 and last - y <= gap:
        if col[y]:
            last = y
        y -= 1
    return last


def track_header_texts(img: Image.Image, top: int, dpi: float, sh: int) -> list[tuple[float, float, str]]:
    """Text of each narrow track's header cell, read upright and rotated both ways: vertical headers ("LITHOLOGY",
    "Interp Lith", "CUTTING %") are only legible one cell at a time. The cells are cut between the vertical rules
    of the header's bottom row and pasted onto one canvas per orientation, so it is one Tesseract call each."""
    W = img.size[0]
    strip = _gray(crop_rgb(img, Box(0, top - sh, W, top)))
    dk = strip < 128
    foot = dk[-max(4, int(0.4 * dpi)) :]
    col = foot.mean(axis=0) >= 0.7
    xs, x = [], 0
    while x < W:  # centres of rule runs
        if col[x]:
            e = x
            while e + 1 < W and col[e + 1]:
                e += 1
            xs.append((x + e) / 2)
            x = e + 1
        else:
            x += 1
    lo, hi = _mm(dpi, 3.0), _mm(dpi, 25.0)
    gap = max(2, int(0.05 * dpi))
    cells = []
    for a, b in zip(xs, xs[1:], strict=False):
        a0, b0 = int(a) + 3, int(b) - 2
        if not lo <= b - a <= hi:
            continue
        y0 = max(_rule_top(dk, int(round(a)), gap), _rule_top(dk, int(round(b)), gap))  # where both side rules start
        if sh - y0 > 0.2 * dpi:
            cells.append((a0, b0, y0))
    if not cells:
        return []
    texts = {c: [] for c in cells}
    up = 2.0 if dpi < 250 else 1.0
    gap = 24
    for k in (0, 1, 3):
        parts = [np.ascontiguousarray(np.rot90(strip[y0:, a0:b0], k)) for a0, b0, y0 in cells]
        if k == 0:  # side by side
            H_ = max(p.shape[0] for p in parts)
            canvas = np.full((H_, sum(p.shape[1] + gap for p in parts)), 255, np.uint8)
            spans, o = [], 0
            for p in parts:
                canvas[H_ - p.shape[0] :, o : o + p.shape[1]] = p
                spans.append((o, o + p.shape[1]))
                o += p.shape[1] + gap
        else:  # stacked rows
            W_ = max(p.shape[1] for p in parts)
            canvas = np.full((sum(p.shape[0] + gap for p in parts), W_), 255, np.uint8)
            spans, o = [], 0
            for p in parts:
                canvas[o : o + p.shape[0], : p.shape[1]] = p
                spans.append((o, o + p.shape[0]))
                o += p.shape[0] + gap
        d = pytesseract.image_to_data(
            Image.fromarray(canvas).resize((round(canvas.shape[1] * up), round(canvas.shape[0] * up))),
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"], strict=False):
            if not t.strip() or float(c) < 30:
                continue
            m = ((x + w / 2) if k == 0 else (y + h / 2)) / up
            for cell, (s0, s1) in zip(cells, spans, strict=False):
                if s0 <= m < s1:
                    texts[cell].append(t.strip().lower())
    return [((a0 + b0) / 2, float(b0 - a0), " ".join(ws)) for (a0, b0, _), ws in texts.items() if ws]


def lith_anchors(img: Image.Image, top: int, dpi: float, words: list) -> list[tuple[float, float, bool]]:
    """x-centre, x-width and strength ("Interp"/"Int." = strong) of 'Lith...' header words or track cells, judged by
    their whole phrase ("% Lithology" and "Lithology Description" do not count)."""
    out = []
    neg = []

    def take(phrases, to_x):
        for ph in phrases:
            text = " ".join(w[4] for w in ph).lower()
            if any(z in text for z in _NEG_HEADER):  # "Cuttings", "% Lith", "Lithology description": not this track
                for w in ph:
                    neg.append(to_x(w))
                continue
            if "lith" not in text or any(z in text for z in _BAD_HEADER):
                continue
            strong = bool(_STRONG_HEADER.search(text))
            for w in ph:
                if "lith" in w[4].lower():
                    out.append((*to_x(w), strong))

    take(_phrases(words), lambda w: ((w[0] + w[2]) / 2, w[2] - w[0]))
    W = img.size[0]
    sh = int(min(top, 2.5 * dpi))
    if sh > 20:
        s = min(1.0, 2000.0 / W)
        strip = _gray(crop_rgb(img, Box(0, top - sh, W, top), scale=s))
        for x, wd, text in track_header_texts(img, top, dpi, sh):
            if any(z in text for z in _NEG_HEADER):
                neg.append((x, wd))
            elif _POS_HEADER.search(text) and not any(z in text for z in _BAD_HEADER):
                out.append((x, -wd, bool(_STRONG_HEADER.search(text))))  # negative width marks a whole cell
        # horizontal headers with raw tokens: the legend word reader drops symbols, so "% Lithology" would otherwise
        # read as "Lithology" (a positive anchor over the percent-lithology track)
        d = pytesseract.image_to_data(Image.fromarray(strip), config="--psm 11", output_type=pytesseract.Output.DICT)
        raw = [
            (x / s, y / s, (x + w) / s, (y + h) / s, t.strip())
            for t, c, x, y, w, h in zip(
                d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"], strict=False
            )
            if t.strip() and float(c) >= 0
        ]
        take(_phrases(raw), lambda w: ((w[0] + w[2]) / 2, w[2] - w[0]))
    # a positive anchor sitting on a negative one is the same header read without its "%" / "Cuttings"
    out = [(x, wd, st) for x, wd, st in out if not any(abs(x - nx) <= max(abs(wd), nwd, 0.1 * dpi) for nx, nwd in neg)]
    lith_anchors.negative = neg  # type: ignore[attr-defined]  # read by detect_boxes right after the call
    return out


def _is_rock_word(t: str) -> bool:
    words = [correct_word(w).lower() for w in re.findall(r"[A-Za-z]{2,}", t)]
    return any(w in ROCK_WORDS and w not in MODIFIER_WORDS for w in words)


def header_words(img: Image.Image, top: int, dpi: float, cfg: LayoutConfig) -> tuple[list, float]:
    """Tesseract words of the header (rows above the body), in page coordinates; and the header scale used."""
    W = img.size[0]
    hh = int(min(top, cfg.header_max_in * dpi))
    if hh < 20:
        return [], 1.0
    s = min(1.0, 2000.0 / W)
    gray = _gray(crop_rgb(img, Box(0, 0, W, hh), scale=s))
    up = 1.5 if dpi * s < 190 else 1.0  # small text at low resolution: upscale for Tesseract
    words = []
    th = max(600, int(0.8 * gray.shape[1]))
    for y in range(0, gray.shape[0], th - 80):
        tile = gray[y : y + th]
        for w in _ocr_words(tile, up, 30):
            cy = y + (w.y0 + w.y1) / 2
            if y > 0 and cy < y + 40:
                continue
            words.append(((w.x0) / s, (y + w.y0) / s, (w.x1) / s, (y + w.y1) / s, w.text))
        if y + th >= gray.shape[0]:
            break
    return words, s


def _rule_rows(prof: np.ndarray, frac: float, gap: int = 3) -> list[int]:
    """Centres of runs of rows (or columns) whose rule coverage is >= frac."""
    out: list[list[int]] = []
    for y in np.where(prof >= frac)[0]:
        if out and y - out[-1][-1] <= gap:
            out[-1].append(int(y))
        else:
            out.append([int(y)])
    return [int(np.mean(r)) for r in out]


def _split_cell(c: tuple, hl: np.ndarray, vl: np.ndarray, s: float, min_side: float, frac: float = 0.9) -> list[tuple]:
    """Sub-cells between internal rules that cross (almost) the whole cell, joining up to 3 neighbours: a legend is
    often one compartment of a bigger frame whose inner divider does not close a rectangle of its own."""
    x0, y0, x1, y1 = (int(round(v * s)) for v in c)
    x1, y1 = min(x1, hl.shape[1]), min(y1, hl.shape[0])
    if x1 - x0 < 2 * min_side or y1 - y0 < min_side:
        return []
    pad = 4
    ys = [y0] + [y0 + pad + y for y in _rule_rows(hl[y0 + pad : y1 - pad, x0:x1].mean(axis=1), frac)] + [y1]
    xs = [x0] + [x0 + pad + x for x in _rule_rows(vl[y0:y1, x0 + pad : x1 - pad].mean(axis=0), frac)] + [x1]
    out = []
    for i in range(len(ys)):
        for j in range(i + 1, min(len(ys), i + 4)):
            for a in range(len(xs)):
                for b in range(a + 1, min(len(xs), a + 4)):
                    if (i, j, a, b) == (0, len(ys) - 1, 0, len(xs) - 1):
                        continue
                    if ys[j] - ys[i] >= min_side and xs[b] - xs[a] >= 2 * min_side:
                        out.append((int(xs[a] / s), int(ys[i] / s), int(xs[b] / s), int(ys[j] / s)))
    return out


def header_cells(img: Image.Image, top: int, dpi: float, cfg: LayoutConfig) -> list[tuple[int, int, int, int]]:
    """Closed rule rectangles in the header, full-width bands between horizontal rules (also cropped to the outermost
    vertical rules crossing them), and the sub-cells of all of these between internal rules."""
    W = img.size[0]
    hh = int(min(top, cfg.header_max_in * dpi))
    if hh < 20:
        return []
    s = min(1.0, 2000.0 / W)
    g = _ink(crop_rgb(img, Box(0, 0, W, hh), scale=s))
    L = max(15, int(0.35 * dpi * s))
    kh, kv = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)), cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))

    def lines(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cv2.morphologyEx(ink, cv2.MORPH_OPEN, kh), cv2.morphologyEx(ink, cv2.MORPH_OPEN, kv)

    hl, vl = lines((g < 160).astype(np.uint8))
    # hairline rules in light grey (PDF exports draw frames at ~200/255): keep them only where they are thin, so a
    # grey fill or a grey swatch is never read as a rule
    hl2, vl2 = lines((g < 230).astype(np.uint8))
    t = max(3, int(round(0.03 * dpi * s)))
    hl |= hl2 & ~cv2.morphologyEx(hl2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, t)))
    vl |= vl2 & ~cv2.morphologyEx(vl2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (t, 1)))
    grid = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells, min_side = [], 0.4 * dpi * s
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 * min_side or h < min_side or w * h > 0.9 * g.size or cv2.contourArea(c) < 0.6 * w * h:
            continue
        cells.append((int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)))
    rowfrac = hl.mean(axis=1)
    lines_y: list[int] = []
    for y in np.where(rowfrac > 0.6)[0]:
        if not lines_y or y - lines_y[-1] > 3:
            lines_y.append(int(y))
    bands = []
    for i in range(len(lines_y)):
        for j in range(i + 1, min(len(lines_y), i + 4)):
            if lines_y[j] - lines_y[i] >= min_side:
                bands.append((0, int(lines_y[i] / s), W, int(lines_y[j] / s)))
    # a band also spans the page margins: crop it to the outermost vertical rules that cross it
    tight = []
    for b in bands:
        y0, y1 = int(b[1] * s) + 4, int(b[3] * s) - 4
        xs = _rule_rows(vl[y0:y1].mean(axis=0), 0.9) if y1 > y0 else []
        if len(xs) >= 2 and xs[-1] - xs[0] > 0.3 * vl.shape[1]:
            tight.append((int(xs[0] / s), b[1], int(xs[-1] / s), b[3]))
    base = list(dict.fromkeys(cells + bands))
    subs = [sc for c in base + tight for sc in _split_cell(c, hl, vl, s, min_side)]
    return list(dict.fromkeys(base + tight + subs))


def _word_cluster(words: list, dpi: float) -> tuple[int, int, int, int] | None:
    hits = np.array([w[:4] for w in words if _is_rock_word(w[4])], float)
    if len(hits) < 3:
        return None
    cy = (hits[:, 1] + hits[:, 3]) / 2
    med = np.median(cy)
    keep = hits[np.abs(cy - med) < max(1.5 * dpi, 3 * np.median(np.abs(cy - med)) + 0.5 * dpi)]
    th = float(np.median(keep[:, 3] - keep[:, 1]))
    return (
        int(max(0, keep[:, 0].min() - 6 * th)),
        int(max(0, keep[:, 1].min() - 2 * th)),
        int(keep[:, 2].max() + 2 * th),
        int(keep[:, 3].max() + 2 * th),
    )


def _snap_cluster(wc: tuple, words: list, cells: list) -> tuple | None:
    """Smallest header cell/band that holds >= 85% of the rock words inside the word-cluster box."""
    x0, y0, x1, y1 = wc
    pts = [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in words if _is_rock_word(w[4])]
    pts = [(x, y) for x, y in pts if x0 <= x <= x1 and y0 <= y <= y1]
    if not pts:
        return None
    best = None
    for c in cells:
        inside = sum(c[0] <= x <= c[2] and c[1] <= y <= c[3] for x, y in pts)
        if inside >= 0.85 * len(pts):
            area = (c[2] - c[0]) * (c[3] - c[1])
            if best is None or area < best[0]:
                best = (area, c)
    return best[1] if best else None


@functools.lru_cache(maxsize=4096)
def _word_kind(t: str) -> str:
    """rock name / lithology modifier / header metadata / other. Metadata wins: "State" (read as slate) or a
    "Cement" field label is not a legend entry."""
    if _META.search(t):
        return "meta"
    words = [correct_word(w).lower() for w in re.findall(r"[A-Za-z]{2,}", t)]
    if any(w in ROCK_WORDS and w not in MODIFIER_WORDS for w in words):
        return "rock"
    if any(w in MODIFIER_WORDS or w in ROCK_WORDS for w in words):
        return "mod"
    return "other"


def legend_score(box: tuple, words: list, page_w: int, header_h: int, cfg: LayoutConfig) -> tuple[float, dict]:
    x0, y0, x1, y1 = box
    kinds = [_word_kind(w[4]) for w in words]
    n_all = sum(k == "rock" for k in kinds) or 1
    n_in = n_rock = n_mod = n_non = 0
    title, t_above = 0.0, None
    for (wx0, wy0, wx1, wy1, t), k in zip(words, kinds, strict=True):
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        n_in += 1
        n_rock += k == "rock"
        n_mod += k == "mod"
        n_non += k == "meta"  # only header metadata counts against a legend candidate
        if _TITLE.search(t) and cy < y0 + 0.35 * (y1 - y0):
            title = 1.0
    # a title printed just above the frame ("Lithology Legend" over a centred legend box) also belongs to it
    if not title:
        for wx0, wy0, wx1, wy1, t in words:
            cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
            h = max(1.0, wy1 - wy0)
            if _TITLE.search(t) and x0 <= cx <= x1 and y0 - 3.0 * h <= cy < y0:
                title, t_above = 1.0, (wy0, h)
                break
    area = (x1 - x0) * (y1 - y0) / float(max(1, page_w * header_h))
    f = {
        "n_rock": n_rock,
        "n_mod": n_mod,
        "share": n_rock / n_all,
        # modifiers ("Dolomitic", "Silty", "Pyrite") are legend vocabulary too: a mineral/modifier panel next to the
        # rock panel is part of the legend, a fossil or drilling-symbol panel is not
        "purity": (n_rock + n_mod) / max(1, n_in),
        "non": n_non / max(1, n_in),
        "title": title,
        "area": area,
    }
    score = (
        1.0 * f["share"]
        + 1.2 * f["purity"]
        + cfg.legend_title_bonus * title
        + 0.03 * min(n_rock, 20)
        - 0.8 * f["non"]
        - 0.4 * area
    )
    if t_above is not None:
        f["title_above_y"], f["title_h"] = t_above
    return score, f


def _rerank_legend(scored: list, dpi: float) -> list:
    """Two ordering rules on top of the score (scored is sorted best first):
    - a candidate nested in the winner, at most half its area, with twice its vocabulary purity and all but at most
      one of its rock words wins: one stray rock word (a "Cement:" field label) must not buy a frame twice the size;
    - when the winner's title sits just above it, the candidate with the same frame that starts just above the title
      wins: the title row belongs to the legend."""
    if not scored:
        return scored
    s0, f0, c0 = scored[0]
    a0 = (c0[2] - c0[0]) * (c0[3] - c0[1])
    tol = 4
    best = None
    for t in scored[1:]:
        _, f, c = t
        inside = c0[0] - tol <= c[0] and c0[1] - tol <= c[1] and c[2] <= c0[2] + tol and c[3] <= c0[3] + tol
        if (
            inside
            and (c[2] - c[0]) * (c[3] - c[1]) <= 0.5 * a0
            and f["n_rock"] >= max(2, f0["n_rock"] - 1)
            and f["purity"] >= 2 * f0["purity"]
        ):
            best = t
            break  # scored is sorted: the first hit is the best-scoring one
    if best is not None:
        scored = [best] + [t for t in scored if t is not best]
    s0, f0, c0 = scored[0]
    if "title_above_y" in f0:
        ty, th, xt = f0["title_above_y"], f0["title_h"], 0.1 * dpi
        ext = [
            t
            for t in scored
            if abs(t[2][0] - c0[0]) <= xt
            and abs(t[2][2] - c0[2]) <= xt
            and abs(t[2][3] - c0[3]) <= xt
            and ty - 2.5 * th <= t[2][1] <= ty
        ]
        if ext:
            e = max(ext, key=lambda t: t[2][1])
            scored = [e] + [t for t in scored if t is not e]
    return scored


def legend_palette(img: Image.Image, box: tuple, n_max: int = 12, noisy: bool = False) -> np.ndarray:
    """Distinct saturated fill colours inside the legend (Lab), used to verify the lithology column. On a noisy scan
    the crop is median-filtered first."""
    x0, y0, x1, y1 = box
    arr = crop_rgb(img, Box(x0, y0, x1, y1), scale=min(1.0, 800.0 / max(1, x1 - x0)))
    if noisy and min(arr.shape[:2]) >= 5:
        arr = cv2.medianBlur(arr, 5)
    arr = arr.reshape(-1, 3).astype(np.int16)
    sat = arr[(arr.max(axis=1) - arr.min(axis=1)) > 40]
    if len(sat) < 50:
        return np.zeros((0, 3), np.float32)
    q = sat // 24
    key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)[:n_max]
    cols = np.array([sat[inv == k].mean(axis=0) for k in order if cnt[k] > 0.004 * len(sat)], np.uint8)
    if len(cols) == 0:
        return np.zeros((0, 3), np.float32)
    return cv2.cvtColor(cols.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Lithology column


def _fits(a: int, b: int, x: float, wd: float) -> bool:
    """Does the header (word of width wd, or whole track cell of width -wd) label the candidate [a, b]?"""
    if wd < 0:  # a track cell labels candidates that cover most of it, not a half-track inside it
        c0, c1 = x + wd / 2, x - wd / 2
        return min(b, c1) - max(a, c0) >= 0.7 * (c1 - c0) and b - a <= 1.3 * (c1 - c0)
    return wd <= 1.5 * (b - a)


def _colour_classes(
    rows_rgb: np.ndarray, key: np.ndarray, sat: np.ndarray, pal: np.ndarray | None, cfg: LayoutConfig
) -> tuple[np.ndarray, int, np.ndarray]:
    """Colour class of every pixel. With a legend palette, each saturated pixel takes the legend colour it is a shade
    of: a black hatch's anti-aliased fringe over a fill is (1 - a) * fill, so it counts as the fill, not as a second
    colour (cost = distance to the best shade s * colour, s >= the darkest fill shade, plus a small bias against dark
    shades so a legend colour that is itself dark wins its own pixels). Legend colours closer than the merge distance
    (Lab) are one class (two shades of one swatch on a scan). Unsaturated pixels (greys) and pages without a palette
    keep the coarse RGB bins of `key` (343 classes). Also returns which saturated pixels match a legend colour or a
    shade of one (within 14 Lab units)."""
    match = np.zeros(key.shape, bool)
    if pal is None or not sat.any():
        return key, 343, match
    P = cv2.cvtColor(pal.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)
    pp = np.maximum((P * P).sum(axis=1), 1.0)
    x = rows_rgb[sat].astype(np.float32)
    step = 100_000  # bound the (pixels x palette) matrices
    near, ok = [], []
    for j in range(0, len(x), step):
        xj = x[j : j + step]
        dot = xj @ P.T
        sh = np.clip(dot / pp, cfg.lith_shade_min, 1.0)
        res = np.sqrt(np.maximum((xj * xj).sum(axis=1, keepdims=True) - 2 * sh * dot + sh * sh * pp, 0.0))
        nj = (res + cfg.lith_shade_bias * 255.0 * (1.0 - sh)).argmin(axis=1)
        near.append(nj)
        shade = np.clip(sh[np.arange(len(nj)), nj, None] * P[nj], 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(np.ascontiguousarray(xj.astype(np.uint8)).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        lab = lab.reshape(-1, 3).astype(np.float32)
        lab_s = cv2.cvtColor(shade.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        d_pal = np.linalg.norm(lab[:, None] - pal[None], axis=2).min(axis=1)
        ok.append((d_pal < 14) | (np.linalg.norm(lab - lab_s, axis=1) < 14))
    dd = np.linalg.norm(pal[:, None] - pal[None], axis=2)
    group = np.arange(len(pal))
    for _ in range(len(pal)):  # connected components of "closer than the merge distance", labelled by smallest index
        group = np.array([group[dd[i] < cfg.lith_palette_merge_de].min() for i in range(len(pal))])
    cls = key.copy()
    cls[sat] = 343 + group[np.concatenate(near)]
    match[sat] = np.concatenate(ok)
    return cls, 343 + len(pal), match


def lith_candidates(
    rows_rgb: np.ndarray,
    edges: list[int],
    dark: np.ndarray,
    dpi: float,
    cfg: LayoutConfig,
    palette: np.ndarray,
    anchors: list[tuple[float, float, bool]],
    rules: list[tuple[int, int]] | None = None,
    neg_anchors: list[tuple[float, float]] | None = None,
) -> list[dict]:
    lo, hi = _mm(dpi, 3.0), _mm(dpi, cfg.max_track_mm)
    s = rows_rgb.astype(np.int16)
    mn, mx = s.min(axis=2), s.max(axis=2)
    nonwhite = (mn < 215) | ((mx - mn) > 35)
    fillpx = nonwhite & (mx >= 90)  # fill colours only: near-black pattern ink is not a colour class
    ink = mx < 128
    key = s // 40
    key = key[..., 0] * 49 + key[..., 1] * 7 + key[..., 2]
    pal = palette if len(palette) else None
    sat = fillpx & ((mx - mn) > 25)
    cls, n_cls, match = _colour_classes(rows_rgb, key, sat, pal, cfg)
    rules = rules or []
    fringe = max(1, round(_mm(dpi, 0.15)))
    out = []
    for i, a in enumerate(edges):
        for b in edges[i + 1 :]:
            w = b - a
            if w < lo:
                continue
            if w > hi:
                break
            # a whole rule strictly inside: two tracks, not one (rules the candidate starts/ends on are fine)
            if any(rs > a + 1 and re_ < b - 1 for rs, re_ in rules):
                continue
            # cues on the interior: a black rule the candidate starts or ends on is not a second colour
            a_, b_ = a, b  # (and the anti-aliased fringe just past it)
            for rs, re_ in rules:
                if rs - 1 <= a <= re_ + 1:
                    a_ = max(a_, re_ + 1 + fringe)
                if rs - 1 <= b <= re_ + 1:
                    b_ = min(b_, rs - fringe)
            if b_ - a_ < 3:
                a_, b_ = a, b
            nw = nonwhite[:, a_:b_]
            fp = fillpx[:, a_:b_]
            fill_r = nw.mean(axis=1)
            k = cls[:, a_:b_]
            R = k.shape[0]
            hist = np.zeros((R, n_cls), np.float32)
            ri = np.repeat(np.arange(R), k.shape[1]).reshape(k.shape)
            np.add.at(hist, (ri[fp], k[fp]), 1.0)
            tot = hist.sum(axis=1)
            ok = (fill_r >= 0.3) & (tot > 0)
            share = hist[ok] / tot[ok, None] if ok.any() else np.zeros((0, n_cls))
            homog = float(np.median(share.max(axis=1))) if len(share) else 0.0
            distinct = float(np.mean((share > 0.08).sum(axis=1))) if len(share) else 9.0
            # down-depth variation: share of rows whose dominant class is the track's most common one (a constant
            # strip scores 1.0 and carries no lithology; an interpreted column changes rock down the page)
            dom = share.argmax(axis=1)
            constant = float(np.bincount(dom).max() / len(dom)) if len(dom) else 1.0
            # colour runs per row (fill pixels only): a percent-lithology track has 2+, an interpreted one 1
            ch = (k[:, 1:] != k[:, :-1]) & fp[:, 1:] & fp[:, :-1]
            runs_row = 1.0 + ch.sum(axis=1)
            runs = float(np.median(runs_row[ok])) if ok.any() else 9.0
            # texture cues (black-and-white pages): hatch-like ink density, same pattern left and right
            inkc = ink[:, a_:b_]
            ink_r = inkc.mean(axis=1)
            tex_fill = float(np.mean((ink_r > 0.05) & (ink_r < 0.6)))
            wi = b_ - a_
            dl = inkc[:, : max(1, wi // 3)].mean(axis=1)
            dr = inkc[:, -max(1, wi // 3) :].mean(axis=1)
            busy = (dl + dr) > 0.04
            lr = float(np.mean(np.abs(dl - dr)[busy] / ((dl + dr)[busy] / 2 + 1e-6))) if busy.any() else 1.0
            thirds = [nw[:, j * wi // 3 : (j + 1) * wi // 3].mean(axis=1) > 0.05 for j in range(3)]
            spread = float(np.mean(thirds[0] & thirds[1] & thirds[2]))
            cov = 0.0
            if pal is not None:  # share of the coloured pixels that are legend colours (or a hatch's shade of one)
                m = match[:, a_:b_][sat[:, a_:b_]]
                if len(m) > 50:
                    cov = float(m.mean())
            outside = np.concatenate([nonwhite[:, max(0, a - 6) : max(0, a - 1)], nonwhite[:, b + 1 : b + 6]], axis=1)
            contrast = float(fill_r.mean() - (outside.mean() if outside.size else 0.0))
            header = float(any(a <= x <= b and _fits(a, b, x, wd) for x, wd, _ in anchors))
            strong = float(any(a <= x <= b and _fits(a, b, x, wd) and st for x, wd, st in anchors))
            # a negative label covers its whole word / track cell: a fragment of the cuttings track inherits it
            cx = (a + b) / 2
            negh = float(any(abs(cx - x) <= wd / 2 + 2 for x, wd in neg_anchors or []))
            out.append(
                {
                    "x0": a,
                    "x1": b,
                    "fill": float((fill_r > 0.6).mean()),
                    "homog": homog,
                    "distinct": distinct,
                    "runs": runs,
                    "constant": constant,
                    "spread": spread,
                    "cov": cov,
                    "contrast": contrast,
                    "tex_fill": tex_fill,
                    "lr": lr,
                    "header": header,
                    "strong_header": strong,
                    "neg_header": negh,
                }
            )
    return out


_FLOOR = {
    "fill": 0.1,
    "homog": 0.05,
    "distinct": 0.2,
    "runs": 0.3,
    "spread": 0.1,
    "contrast": 0.1,
    "cov": 0.1,
    "tex_fill": 0.1,
    "lr": 0.1,
}


_ANCHORED_FIRST = True


def _dedupe_tracks(cands: list[dict], thr: float = 0.85) -> list[dict]:
    """Keep one of each group of near-identical candidates (x-IoU > thr), so a dozen variants of one track do not
    become the pool's median and hide it from the robust scores."""
    order = sorted(cands, key=lambda c: (-c["contrast"], -c["fill"]))
    kept: list[dict] = []
    for c in order:
        if all(
            (min(c["x1"], k["x1"]) - max(c["x0"], k["x0"])) / (max(c["x1"], k["x1"]) - min(c["x0"], k["x0"])) <= thr
            for k in kept
        ):
            kept.append(c)
    return kept


def _bilevel(rows_rgb: np.ndarray) -> bool:
    """True when the ink is black on white (a B/W scan or render): few mid-grey pixels among the non-white ones."""
    g = _gray(rows_rgb[::2, ::2])
    nonwhite = g < 230
    return bool(((g >= 60) & nonwhite).sum() < 0.3 * max(1, nonwhite.sum()))


def _page_is_grey(rows_rgb: np.ndarray) -> bool:
    """True when the page carries (almost) no colour: a scan or a greyscale render."""
    s = rows_rgb[::4, ::4].astype(np.int16)
    return bool(((s.max(axis=2) - s.min(axis=2)) > 40).mean() < 0.002)


def _rank_sum(cands: list[dict], cues: dict[str, float]) -> None:
    """Score = sum of robust within-page scores (v - median) / max(1.48 MAD, floor) with fixed signs (no training).
    The floor stops tiny differences (0.924 vs 0.917) from being blown up to +-1 on small candidate pools."""
    for k, sign in cues.items():
        v = np.array([c[k] for c in cands], float)
        med = np.median(v)
        scale = max(1.48 * float(np.median(np.abs(v - med))), _FLOOR.get(k, 0.1))
        for c, x in zip(cands, v, strict=False):
            c["score"] = c.get("score", 0.0) + sign * float(np.clip((x - med) / scale, -3.0, 3.0))


# ---------------------------------------------------------------------------------------------------------------
# Main


def detect_boxes(img: Image.Image, cfg: LayoutConfig) -> LayoutResult:
    W, H = img.size
    dpi = image_dpi(img) or W / 8.5
    sigma = _noise_sigma(img)
    noisy = sigma > 3.0  # a noisy scan: colour cues are read on median-filtered pixels
    top, bot, _ = body_range(img, dpi, cfg)
    rows = _straight(_rows(img, np.linspace(top + 0.05 * (bot - top), bot - 0.02 * (bot - top), 500).astype(int)))
    edges, dark, rules = vertical_edges(rows, dpi)
    diag: dict = {"dpi": dpi, "body": [top, bot], "n_edges": len(edges), "noise": round(sigma, 1)}

    # depth
    toks, rots = number_tokens(img, top, bot, dpi, cfg)
    diag["depth_rotations"] = rots
    dcs = depth_candidates(toks, edges, dark, dpi, cfg, top, bot, W)
    if not dcs or dcs[0]["regular_only"] or dcs[0]["inliers"] < 4:
        # labels can be further apart than a band is tall (5 in/100 ft logs): read taller, interleaved bands too
        extra = np.linspace(0.12, 0.88, cfg.depth_bands) + 0.38 / max(1, cfg.depth_bands - 1)
        more, _ = number_tokens(
            img, top, bot, dpi, cfg, fracs=np.clip(extra, 0.05, 0.95), band_in=2 * cfg.depth_band_in, rots=rots
        )
        toks += more
        dcs = depth_candidates(toks, edges, dark, dpi, cfg, top, bot, W)
        diag["depth_second_pass"] = True
    diag["depth"] = [{k: v for k, v in d.items()} for d in dcs[:3]]
    diag["depth_all"] = [list(d["box"]) for d in dcs[:10]]
    if dcs:
        depth = dcs[0]["box"]
        # share of the cluster's labels on one line is the strongest single cue (dev AUROC 0.90)
        d0 = dcs[0]
        frac = d0["inliers"] / max(1, d0["n"])
        d_conf = float(np.clip(frac * min(1.0, d0["inliers"] / 6.0), 0, 1) * (0.7 if d0["regular_only"] else 1.0))
    else:
        depth, d_conf = (0, top, max(1, int(0.05 * W)), bot), 0.0

    # legend
    words, _ = header_words(img, top, dpi, cfg)
    hh = int(min(top, cfg.header_max_in * dpi))
    cands = header_cells(img, top, dpi, cfg)
    wc = _word_cluster(words, dpi)
    unsnapped = None
    cands = list(dict.fromkeys(map(tuple, cands)))
    if wc:
        snapped = _snap_cluster(wc, words, cands)
        if snapped:
            cands.append(snapped)
        else:  # no frame holds it: keep the raw cluster only as a penalised last resort (it often spans accessories)
            cands.append(wc)
            unsnapped = wc
    scored = []
    for c in cands:
        sc, f = legend_score(c, words, W, max(1, hh), cfg)
        if unsnapped is not None and tuple(c) == tuple(unsnapped):
            sc -= cfg.legend_unsnapped_penalty
            f["unsnapped"] = 1.0
        scored.append((sc, f, c))
    scored.sort(key=lambda t: -t[0])
    scored = _rerank_legend(scored, dpi)
    diag["legend"] = [
        {"box": list(c), "score": round(sc, 3), **{k: round(v, 3) for k, v in f.items()}} for sc, f, c in scored[:3]
    ]
    diag["legend_all"] = [list(c) for _, _, c in scored[:10]]
    if scored:
        legend = scored[0][2]
        f0 = scored[0][1]  # a pure rock-word frame (little other text, not a page-sized band) is a real legend
        l_conf = float(np.clip(0.25 + 1.5 * f0["purity"] - f0["non"] - 0.5 * f0["area"], 0.0, 1.0))
    else:
        legend, l_conf = (0, 0, W, max(1, min(top, H))), 0.0

    # lithology
    anchors = lith_anchors(img, top, dpi, words)
    neg_anchors = getattr(lith_anchors, "negative", [])
    diag["neg_anchors"] = [[round(x), round(wd)] for x, wd in neg_anchors]
    diag["anchors"] = [[round(x), round(wd), int(st)] for x, wd, st in anchors]
    palette = legend_palette(img, legend, noisy=noisy) if img.mode not in ("1", "L") else np.zeros((0, 3), np.float32)
    # hatch-texture path only for bilevel pages (black ink on white); a colour log converted to grey keeps its solid
    # mid-grey fills, and the colour cues work on those grey levels
    bw_page = (img.mode in ("1", "L") or (not len(palette) and _page_is_grey(rows))) and _bilevel(rows)
    lrows = rows
    if bw_page:  # hatching skips rows: measure texture on thin bands (darkest pixel over ~0.03 in of rows)
        band = max(2, int(0.03 * dpi))
        ys = np.linspace(top + 0.05 * (bot - top), bot - 0.02 * (bot - top) - band, 300).astype(int)
        lrows = _straight(np.stack([_rows(img, np.arange(y, y + band)).min(axis=0) for y in ys]))
    if noisy and not bw_page:  # colour cues on denoised rows (rules are found on the raw rows)
        lrows = _median_x(lrows)
    lcs = lith_candidates(lrows, edges, dark, dpi, cfg, palette, anchors, rules, neg_anchors)
    if dcs:  # never let the depth labels sit inside a lithology candidate (touching its edge is fine)
        lxc = sum(dcs[0]["labels"]) / 2
        lcs = [c for c in lcs if not (c["x0"] < lxc < c["x1"])]
    lcs = _dedupe_tracks(lcs)
    bw = bw_page
    diag["bw"] = bool(bw)
    if bw:  # colour cues are constant on black-and-white pages: judge the hatch texture instead
        # single sampled rows often fall in the white gap between hatch rows, so the fill gate is loose and the
        # decisive cue is "the same texture across the width" (a cuttings track has two patterns side by side)
        gated = [c for c in lcs if c["tex_fill"] >= 0.15]
        cues = {"lr": -2.0, "spread": 0.5, "contrast": 0.5}  # texture fill only gates: dotted grid tracks fill too
    else:
        # colours per row is a ranking cue, not a gate: one borderline cue (e.g. a hatch over the fill) must not
        # remove the right track from the pool when every other cue favours it
        gated = [c for c in lcs if c["fill"] >= 0.3 and (not len(palette) or c["cov"] >= 0.4)]
        cues = {"fill": 1.0, "homog": 1.0, "distinct": -1.0, "runs": -1.0, "spread": 1.0, "contrast": 0.5}
        if len(palette):
            cues["cov"] = 1.0
    if not gated and not bw:  # nothing passes: relax the gates before falling back to every candidate
        gated = [c for c in lcs if c["fill"] >= 0.1 and (not len(palette) or c["cov"] >= 0.2)]
    if not gated and not bw and len(palette):  # sparse patterns on white (e.g. "X X" igneous): legend colours decide
        gated = [c for c in lcs if c["cov"] >= 0.5 and c["fill"] >= 0.05]
    if not gated:  # last resort: anything but a blank strip
        gated = [c for c in lcs if c["fill"] >= 0.05 or c["tex_fill"] >= 0.15]
    if gated:  # the track under a "Lithology" header always competes, even if a cue gate rejected it
        gated += [c for c in lcs if c["header"] and c not in gated]
    if any(not c["neg_header"] for c in gated):  # a track under a cuttings/% header is never the interpreted one
        gated = [c for c in gated if not c["neg_header"]]
    pool = gated or lcs
    if pool:
        _rank_sum(pool, cues)
        for c in pool:
            c["score"] += cfg.lith_header_bonus * c["header"] - cfg.lith_negative_header_penalty * c["neg_header"]
            if not bw:  # one colour all the way down carries no lithology: a soft penalty (a one-rock log still wins)
                k = (c["constant"] - cfg.lith_constant_share) / max(1e-6, 1.0 - cfg.lith_constant_share)
                c["score"] -= cfg.lith_constant_penalty * float(np.clip(k, 0.0, 1.0))
        pool.sort(key=lambda c: -c["score"])
        best = pool[0]
        # a track under a "Lithology"/"Interp" header (and under no cuttings/% header) is chosen first: the header is
        # the most reliable cue we have; the cue score only ranks tracks within that group
        # only a strong name ("Interp Lith", "Int. Lith") overrides the cues; a bare "Lith" only adds its bonus
        anchored = [c for c in pool if c["strong_header"] and not c["neg_header"] and (bw or c["fill"] >= 0.3)]
        anchored = anchored if _ANCHORED_FIRST else []
        if anchored:
            best = anchored[0]
            pool = [best] + [c for c in pool if c is not best]
        diag["lith_anchored"] = bool(anchored)
        lith = (best["x0"], top, best["x1"], bot)
        # a "Lithology" header over the pick is the strongest cue (dev AUROC 0.81); then a clear margin and, on colour
        # pages, one colour class per row
        margin = best["score"] - (pool[1]["score"] if len(pool) > 1 else 0.0)
        one_class = 1.0 if (bw or best["distinct"] <= 1.15) else 0.0
        li_conf = 0.3 + 0.35 * best["header"] + 0.2 * float(np.clip(margin / 3.0, 0, 1)) + 0.15 * one_class
        li_conf = float(np.clip(li_conf, 0.0, 1.0)) * (1.0 if gated else 0.5)
        diag["lith"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in pool[:3]]
        diag["lith_all"] = [[c["x0"], top, c["x1"], bot] for c in pool[:10]]
    else:
        lith, li_conf = (int(0.5 * W), top, int(0.55 * W), bot), 0.0

    # joint resolution: if the depth box overlaps the chosen lithology column, shrink it to its labels' side
    if dcs and pool and min(lith[2], depth[2]) - max(lith[0], depth[0]) > 2:
        lx0, lx1 = dcs[0]["labels"]
        if (lith[0] + lith[2]) / 2 >= (lx0 + lx1) / 2:  # lithology to the right of the labels
            new = (depth[0], depth[1], max(lx1, min(depth[2], lith[0])), depth[3])
        else:
            new = (min(lx0, max(depth[0], lith[2])), depth[1], depth[2], depth[3])
        if new[2] - new[0] >= _mm(dpi, cfg.min_track_mm):  # never produce an empty or inverted box
            depth = new
        diag["depth_shrunk"] = list(depth)

    def box(t: tuple) -> Box:
        x0, y0, x1, y1 = (int(v) for v in t)
        return Box(max(0, x0), max(0, y0), max(x0 + 1, min(W, x1)), max(y0 + 1, min(H, y1)))

    boxes = DocumentBoxes(depth_column=box(depth), lithology_column=box(lith), legend=box(legend))
    conf = {
        "depth_column": round(d_conf, 3),
        "interpreted_lithology_column": round(li_conf, 3),
        "lithology_legend": round(l_conf, 3),
    }
    return LayoutResult(boxes, conf, diag)
