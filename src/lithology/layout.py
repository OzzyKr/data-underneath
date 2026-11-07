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


def _rows(img: Image.Image, ys: np.ndarray, x0: int = 0, x1: int | None = None) -> np.ndarray:
    W = img.size[0]
    x1 = W if x1 is None else x1
    return np.stack([np.asarray(img.crop((x0, int(y), x1, int(y) + 1)).convert("RGB"))[0] for y in ys])


def _mm(dpi: float, mm: float) -> float:
    return mm / 25.4 * dpi


# ---------------------------------------------------------------------------------------------------------------
# Body


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
    win_h = int(min(bot - top, cfg.header_search_in * dpi))
    if win_h > 10:
        g = _ink(crop_rgb(img, Box(xa, top, xb, top + win_h))) < 150
        cand = np.where(g.mean(axis=1) > 0.85)[0]
        if len(cand):
            top = top + int(cand[-1]) + 1
    return top, bot, xs


def estimate_skew(dark: np.ndarray, ys: np.ndarray, W: int, n_bands: int = 8) -> tuple[float, float]:
    """Horizontal drift of the vertical rules down the page (scan skew), as x_shift(y) = a*y + b relative to the
    page middle: each band's dark-column profile is cross-correlated with the middle band's."""
    n = len(ys)
    if n < 4 * n_bands:
        return 0.0, 0.0
    # only the lower 60% of the page (always log body; the header's structure differs and spoils the fit)
    edges = np.linspace(int(0.4 * n), n, n_bands + 1).astype(int)
    prof = [dark[a:b].mean(axis=0).astype(np.float32) for a, b in zip(edges[:-1], edges[1:], strict=False)]
    ref = prof[n_bands // 2] - prof[n_bands // 2].mean()  # middle of the analysed part
    if ref.std() < 1e-3:
        return 0.0, 0.0
    maxs = max(4, int(0.03 * W))
    yc, sh = [], []
    for (a, b), p in zip(zip(edges[:-1], edges[1:], strict=False), prof, strict=False):
        q = p - p.mean()
        if q.std() < 1e-3:
            continue
        cc = [float(np.dot(np.roll(q, -k), ref)) for k in range(-maxs, maxs + 1)]
        yc.append(float(ys[(a + b) // 2]))
        sh.append(float(np.argmax(cc) - maxs))
    if len(sh) < 3:
        return 0.0, 0.0
    a, b = np.polyfit(yc, sh, 1)
    resid = np.abs(np.polyval([a, b], yc) - np.array(sh))
    if np.median(resid) > 3:  # inconsistent shifts: not a clean skew, leave the rows alone
        return 0.0, 0.0
    return float(a), float(b)


def deskew_rows(dark: np.ndarray, ys: np.ndarray, skew: tuple[float, float]) -> np.ndarray:
    """Shift each sampled row back by the estimated drift (no-op when the drift is under 3 px over the page)."""
    a, b = skew
    if abs(a) * (ys[-1] - ys[0]) < 3:
        return dark
    out = np.empty_like(dark)
    for i, y in enumerate(ys):
        out[i] = np.roll(dark[i], -int(round(a * y + b)))
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
        lo = int(max(0, xl - reach))
        left = np.where(dark[lo : int(xl)] > 0.6)[0]
        x0 = lo + int(left.max()) + 1 if len(left) else int(max(0, xl - 0.1 * dpi))
        hi = int(min(W, xr + reach))
        right = np.where(dark[int(xr) + 1 : hi] > 0.6)[0]
        x1 = int(xr) + 1 + int(right.min()) if len(right) else int(min(W, xr + 0.1 * dpi))
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


def lith_anchors(img: Image.Image, top: int, dpi: float, words: list) -> list[tuple[float, float]]:
    """x-centre and x-width of 'Lith...' header words, judged by their whole phrase ("% Lithology" and
    "Lithology Description" do not count). Vertical track headers are read from the strip above the body."""
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
            for w in ph:
                if "lith" in w[4].lower():
                    out.append(to_x(w))

    take(_phrases(words), lambda w: ((w[0] + w[2]) / 2, w[2] - w[0]))
    W = img.size[0]
    sh = int(min(top, 2.5 * dpi))
    if sh > 20:
        s = min(1.0, 2000.0 / W)
        strip = _gray(crop_rgb(img, Box(0, top - sh, W, top), scale=s))
        hs, ws_ = strip.shape
        for k in (1, 3):  # np.rot90: 1 = counter-clockwise, 3 = clockwise
            rot = np.ascontiguousarray(np.rot90(strip, k))
            rw = [(w.x0, w.y0, w.x1, w.y1, w.text) for w in _ocr_words(rot, 1.5 if dpi * s < 190 else 1.0, 30)]
            if k == 1:  # rotated (u=col, v=row) -> original x = ws-1-v ; width in x = word height
                take(_phrases(rw), lambda w: ((ws_ - 1 - (w[1] + w[3]) / 2) / s, (w[3] - w[1]) / s))
            else:  # k == 3: original x = v
                take(_phrases(rw), lambda w: (((w[1] + w[3]) / 2) / s, (w[3] - w[1]) / s))
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
    out = [(x, wd) for x, wd in out if not any(abs(x - nx) <= max(wd, nwd, 0.1 * dpi) for nx, nwd in neg)]
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


def header_cells(img: Image.Image, top: int, dpi: float, cfg: LayoutConfig) -> list[tuple[int, int, int, int]]:
    """Closed rule rectangles in the header plus full-width bands between horizontal rules."""
    W = img.size[0]
    hh = int(min(top, cfg.header_max_in * dpi))
    if hh < 20:
        return []
    s = min(1.0, 2000.0 / W)
    g = _ink(crop_rgb(img, Box(0, 0, W, hh), scale=s))
    ink = (g < 160).astype(np.uint8)
    L = max(15, int(0.35 * dpi * s))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    grid = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells, min_side = [], 0.4 * dpi * s
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 * min_side or h < min_side or w * h > 0.9 * g.size or cv2.contourArea(c) < 0.6 * w * h:
            continue
        cells.append((int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)))
    rowfrac = hl.mean(axis=1)
    lines: list[int] = []
    for y in np.where(rowfrac > 0.6)[0]:
        if not lines or y - lines[-1] > 3:
            lines.append(int(y))
    for i in range(len(lines)):
        for j in range(i + 1, min(len(lines), i + 4)):
            if lines[j] - lines[i] >= min_side:
                cells.append((0, int(lines[i] / s), W, int(lines[j] / s)))
    return cells


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


def legend_score(box: tuple, words: list, page_w: int, header_h: int, cfg: LayoutConfig) -> tuple[float, dict]:
    x0, y0, x1, y1 = box
    n_all = sum(_is_rock_word(w[4]) for w in words) or 1
    n_in = n_rock = n_non = 0
    title = 0.0
    for wx0, wy0, wx1, wy1, t in words:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        n_in += 1
        rock = _is_rock_word(t)
        n_rock += rock
        if not rock and _META.search(t):  # only header metadata counts against a legend candidate
            n_non += 1
        if _TITLE.search(t) and cy < y0 + 0.35 * (y1 - y0):
            title = 1.0
    # a title printed just above the frame ("Lithology Legend" over a centred legend box) also belongs to it
    if not title:
        for wx0, wy0, wx1, wy1, t in words:
            cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
            h = max(1.0, wy1 - wy0)
            if _TITLE.search(t) and x0 <= cx <= x1 and y0 - 3.0 * h <= cy < y0:
                title = 1.0
                break
    area = (x1 - x0) * (y1 - y0) / float(max(1, page_w * header_h))
    f = {
        "n_rock": n_rock,
        "share": n_rock / n_all,
        "purity": n_rock / max(1, n_in),
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
    return score, f


def legend_palette(img: Image.Image, box: tuple, n_max: int = 12) -> np.ndarray:
    """Distinct saturated fill colours inside the legend (Lab), used to verify the lithology column."""
    x0, y0, x1, y1 = box
    arr = crop_rgb(img, Box(x0, y0, x1, y1), scale=min(1.0, 800.0 / max(1, x1 - x0))).reshape(-1, 3).astype(np.int16)
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


def lith_candidates(
    rows_rgb: np.ndarray,
    edges: list[int],
    dark: np.ndarray,
    dpi: float,
    cfg: LayoutConfig,
    palette: np.ndarray,
    anchors: list[tuple[float, float]],
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
    rules = rules or []
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
            nw = nonwhite[:, a:b]
            fp = fillpx[:, a:b]
            fill_r = nw.mean(axis=1)
            k = key[:, a:b]
            R = k.shape[0]
            hist = np.zeros((R, 343), np.float32)
            ri = np.repeat(np.arange(R), k.shape[1]).reshape(k.shape)
            np.add.at(hist, (ri[fp], k[fp]), 1.0)
            tot = hist.sum(axis=1)
            ok = (fill_r >= 0.3) & (tot > 0)
            share = hist[ok] / tot[ok, None] if ok.any() else np.zeros((0, 343))
            homog = float(np.median(share.max(axis=1))) if len(share) else 0.0
            distinct = float(np.mean((share > 0.08).sum(axis=1))) if len(share) else 9.0
            # colour runs per row (fill pixels only): a percent-lithology track has 2+, an interpreted one 1
            ch = (k[:, 1:] != k[:, :-1]) & fp[:, 1:] & fp[:, :-1]
            runs_row = 1.0 + ch.sum(axis=1)
            runs = float(np.median(runs_row[ok])) if ok.any() else 9.0
            # texture cues (black-and-white pages): hatch-like ink density, same pattern left and right
            inkc = ink[:, a:b]
            ink_r = inkc.mean(axis=1)
            tex_fill = float(np.mean((ink_r > 0.05) & (ink_r < 0.6)))
            dl = inkc[:, : max(1, w // 3)].mean(axis=1)
            dr = inkc[:, -max(1, w // 3) :].mean(axis=1)
            busy = (dl + dr) > 0.04
            lr = float(np.mean(np.abs(dl - dr)[busy] / ((dl + dr)[busy] / 2 + 1e-6))) if busy.any() else 1.0
            thirds = [nw[:, j * w // 3 : (j + 1) * w // 3].mean(axis=1) > 0.05 for j in range(3)]
            spread = float(np.mean(thirds[0] & thirds[1] & thirds[2]))
            cov = 0.0
            if pal is not None:
                px = rows_rgb[:, a:b][fp & ((mx[:, a:b] - mn[:, a:b]) > 25)]
                if len(px) > 50:
                    lab = cv2.cvtColor(px[:: max(1, len(px) // 3000)].reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
                    lab = lab.reshape(-1, 3).astype(np.float32)
                    cov = float((np.linalg.norm(lab[:, None] - pal[None], axis=2).min(axis=1) < 14).mean())
            outside = np.concatenate([nonwhite[:, max(0, a - 6) : max(0, a - 1)], nonwhite[:, b + 1 : b + 6]], axis=1)
            contrast = float(fill_r.mean() - (outside.mean() if outside.size else 0.0))
            header = float(any(a <= x <= b and wd <= 1.5 * w for x, wd in anchors))
            negh = float(any(a <= x <= b and wd <= 1.5 * w for x, wd in neg_anchors or []))
            out.append(
                {
                    "x0": a,
                    "x1": b,
                    "fill": float((fill_r > 0.6).mean()),
                    "homog": homog,
                    "distinct": distinct,
                    "runs": runs,
                    "spread": spread,
                    "cov": cov,
                    "contrast": contrast,
                    "tex_fill": tex_fill,
                    "lr": lr,
                    "header": header,
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
    top, bot, _ = body_range(img, dpi, cfg)
    rows = _rows(img, np.linspace(top + 0.05 * (bot - top), bot - 0.02 * (bot - top), 500).astype(int))
    edges, dark, rules = vertical_edges(rows, dpi)
    diag: dict = {"dpi": dpi, "body": [top, bot], "n_edges": len(edges)}

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
    diag["anchors"] = [[round(x), round(wd)] for x, wd in anchors]
    palette = legend_palette(img, legend) if img.mode not in ("1", "L") else np.zeros((0, 3), np.float32)
    bw_page = img.mode in ("1", "L") or (not len(palette) and _page_is_grey(rows))
    lrows = rows
    if bw_page:  # hatching skips rows: measure texture on thin bands (darkest pixel over ~0.03 in of rows)
        band = max(2, int(0.03 * dpi))
        ys = np.linspace(top + 0.05 * (bot - top), bot - 0.02 * (bot - top) - band, 300).astype(int)
        lrows = np.stack([_rows(img, np.arange(y, y + band)).min(axis=0) for y in ys])
    lcs = lith_candidates(lrows, edges, dark, dpi, cfg, palette, anchors, rules, neg_anchors)
    if dcs:  # never let a depth candidate's labels sit inside a lithology candidate
        lx = dcs[0]["labels"]
        lcs = [c for c in lcs if min(c["x1"], lx[1]) - max(c["x0"], lx[0]) <= 2]
    bw = bw_page
    diag["bw"] = bool(bw)
    if bw:  # colour cues are constant on black-and-white pages: judge the hatch texture instead
        # single sampled rows often fall in the white gap between hatch rows, so the fill gate is loose and the
        # decisive cue is "the same texture across the width" (a cuttings track has two patterns side by side)
        gated = [c for c in lcs if c["tex_fill"] >= 0.15]
        cues = {"lr": -2.0, "spread": 0.5, "contrast": 0.5}  # texture fill only gates: dotted grid tracks fill too
    else:
        gated = [c for c in lcs if c["distinct"] <= 1.3 and c["fill"] >= 0.3 and (not len(palette) or c["cov"] >= 0.4)]
        cues = {"fill": 1.0, "homog": 1.0, "distinct": -1.0, "runs": -1.0, "spread": 1.0, "contrast": 0.5}
        if len(palette):
            cues["cov"] = 1.0
    if not gated and not bw:  # nothing passes: relax the gates before falling back to every candidate
        gated = [c for c in lcs if c["distinct"] <= 1.6 and c["fill"] >= 0.1 and (not len(palette) or c["cov"] >= 0.2)]
    if not gated and not bw and len(palette):  # sparse patterns on white (e.g. "X X" igneous): legend colours decide
        gated = [c for c in lcs if c["cov"] >= 0.5 and c["fill"] >= 0.05]
    if not gated:  # last resort: anything but a blank strip
        gated = [c for c in lcs if c["fill"] >= 0.05 or c["tex_fill"] >= 0.15]
    if gated:  # the track under a "Lithology" header always competes, even if a cue gate rejected it
        gated += [c for c in lcs if c["header"] and c not in gated]
    pool = gated or lcs
    if pool:
        _rank_sum(pool, cues)
        for c in pool:
            c["score"] += cfg.lith_header_bonus * c["header"] - cfg.lith_negative_header_penalty * c["neg_header"]
        pool.sort(key=lambda c: -c["score"])
        best = pool[0]
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
        if lith[0] >= lx1 - 2:  # lithology to the right of the labels
            depth = (depth[0], depth[1], min(depth[2], lith[0]), depth[3])
        else:
            depth = (max(depth[0], lith[2]), depth[1], depth[2], depth[3])
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
