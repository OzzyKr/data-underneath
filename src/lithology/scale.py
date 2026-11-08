"""Depth scale: read the printed depth labels with OCR and fit a robust pixel-to-depth mapping.

The filename scale times the TIFF DPI gives an independent pixels-per-foot check. Without OCR labels the
scale is reported as missing, not invented.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image

from lithology.config import ScaleConfig
from lithology.io import crop_rgb, image_dpi, scale_inches_per_100ft
from lithology.models import Box, DepthScale

log = logging.getLogger(__name__)

_TESS_CONFIG = "--psm 11 -c tessedit_char_whitelist=0123456789,."
_INTEGER = re.compile(r"^\d{1,3}(,\d{3})+$|^\d+$")  # 12345 or 12,345 ; decimals are a secondary (TVD) scale


@dataclass(frozen=True)
class DepthLabel:
    """One hypothesis for one OCR token. Tokens can carry several hypotheses (same ``token``)."""

    y: float  # full-image row of the token's centre
    depth: float
    conf: float
    token: int
    primary: bool = True  # False for the "first character dropped" hypothesis


def prior_px_per_ft(img: Image.Image, doc_id: str) -> float | None:
    """Pixels per foot from filename scale x DPI, or None when either is unknown."""
    inches = scale_inches_per_100ft(doc_id)
    dpi = image_dpi(img)
    if inches is None or dpi is None:
        return None
    return dpi * inches / 100.0


def hypotheses(text: str, max_depth: float) -> list[tuple[float, bool]]:
    """Depth values a token may stand for.

    A tick mark touching the first digit is often read as an extra leading '1' or '7', so the value without its
    first character is a second hypothesis. The fit decides which (if either) is consistent with the others.
    """
    text = text.strip().strip(",")  # a trailing comma is a common misread of tick ink
    if not _INTEGER.match(text):
        return []
    digits = text.replace(",", "")
    out: dict[float, bool] = {}
    for cand, primary in ((digits, True), (digits[1:], False)):
        if 2 <= len(cand) <= 5 and not (len(cand) > 1 and cand[0] == "0"):
            v = float(cand)
            if 0 < v <= max_depth and v not in out:
                out[v] = primary
    return list(out.items())


def _ocr_window(
    gray: np.ndarray,
    k: int,
    upscale: float,
    y_origin: int,
    first_token: int,
    max_depth: float,
    min_conf: float = 30,
) -> list[DepthLabel]:
    """OCR one window rotated by k*90 degrees; return hypotheses with centres mapped back to full-image rows."""
    h = gray.shape[0]
    pil = Image.fromarray(np.ascontiguousarray(np.rot90(gray, k)))
    if upscale != 1.0:
        pil = pil.resize((round(pil.width * upscale), round(pil.height * upscale)), Image.Resampling.LANCZOS)
    data = pytesseract.image_to_data(pil, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT)
    out: list[DepthLabel] = []
    token = first_token
    for text, conf, left, top, width, height in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"], strict=False
    ):
        if float(conf) < min_conf:
            continue
        values = hypotheses(text, max_depth)
        if not values:
            continue
        cx = (left + width / 2) / upscale
        cy = (top + height / 2) / upscale
        y_local = cy if k == 0 else (cx if k == 1 else h - 1 - cx)  # undo the rotation
        out += [DepthLabel(y=y_origin + y_local, depth=v, conf=float(conf), token=token, primary=p) for v, p in values]
        token += 1
    return out


def fit_line(
    labels: list[DepthLabel], residual_ft: float, max_candidates: int = 80
) -> tuple[float, float, list[DepthLabel]] | None:
    """Exhaustive (deterministic) RANSAC over hypothesis pairs from different tokens, then least squares.

    Each token contributes at most one inlier (its closest hypothesis). Depth must increase downwards.
    Returns (ft_per_px, depth_at_y0, inlier labels) or None.
    """
    if len({lb.token for lb in labels}) < 2:
        return None
    y = np.array([lb.y for lb in labels])
    d = np.array([lb.depth for lb in labels])
    tok = np.array([lb.token for lb in labels])
    prim = np.array([lb.primary for lb in labels])
    # candidate pairs come from an evenly spread subset; every hypothesis is still scored against each line
    order = np.argsort(y)
    cand = order[np.linspace(0, len(order) - 1, min(len(order), max_candidates)).round().astype(int)]
    best_key: tuple[float, float] | None = None
    best_line: tuple[float, float] | None = None  # (slope, intercept)
    for a_pos, i in enumerate(cand):
        for j in cand[a_pos + 1 :]:
            if tok[i] == tok[j] or abs(y[j] - y[i]) < 1:
                continue
            s = (d[j] - d[i]) / (y[j] - y[i])
            if s <= 0:
                continue
            res = np.abs(d - (d[i] + s * (y - y[i])))
            ok = res <= residual_ft
            # Tokens explained, where an explanation needing a dropped digit counts half; then tightest fit.
            explained: dict[int, float] = {}
            for t, p in zip(tok[ok], prim[ok], strict=True):
                explained[int(t)] = max(explained.get(int(t), 0.0), 1.0 if p else 0.5)
            key = (sum(explained.values()), -float(np.median(res[ok])))
            if best_key is None or key > best_key:
                best_key, best_line = key, (s, d[i] - s * y[i])
    if best_line is None or best_key is None or best_key[0] < 1.5:
        return None
    s, b0 = best_line
    res = np.abs(d - (s * y + b0))
    chosen: dict[int, int] = {}  # token -> index of its closest inlier hypothesis
    for idx in np.flatnonzero(res <= residual_ft):
        t = int(tok[idx])
        if t not in chosen or res[idx] < res[chosen[t]]:
            chosen[t] = int(idx)
    idx = np.array(sorted(chosen.values()))
    A = np.vstack([y[idx], np.ones(len(idx))]).T
    (s, b), *_ = np.linalg.lstsq(A, d[idx], rcond=None)
    if s <= 0:
        return None
    return float(s), float(b), [labels[k] for k in idx]


def read_depth_labels(img: Image.Image, dc: Box, doc_id: str, cfg: ScaleConfig) -> list[DepthLabel]:
    """OCR windows spread down the depth column, in whichever orientation the labels are printed."""
    prior = prior_px_per_ft(img, doc_id)
    win = round(cfg.window_ft * prior) if prior else cfg.window_px_fallback
    win = int(np.clip(win, 800, 4000))
    dpi = image_dpi(img) or 200.0
    upscale = 2.0 if dpi < 180 else 1.0
    pad = max(4, dc.width // 10)
    col = Box(max(0, dc.x0 - pad), dc.y0, min(img.width, dc.x1 + pad), dc.y1)
    n_windows = max(1, min(cfg.n_windows, -(-col.height // win)))  # short columns are covered end to end
    starts = np.linspace(col.y0, max(col.y0, col.y1 - win), n_windows).round().astype(int).tolist()

    def window(y0: int) -> np.ndarray:
        rgb = crop_rgb(img, Box(col.x0, y0, col.x1, min(col.y1, y0 + win)))
        return (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)

    # orientation: OCR a few probe windows each way and keep the one whose labels best fit one line
    probes = starts[:: max(1, len(starts) // 3)][:3]
    probe_gray = {s: window(s) for s in probes}
    best_k, labels, best_score = 0, [], (-1, -1)
    for k in (0, 1, -1):
        found: list[DepthLabel] = []
        for s in probes:
            nxt = max((lb.token for lb in found), default=-1) + 1
            found += _ocr_window(probe_gray[s], k, upscale, s, nxt, cfg.max_depth_ft)
        fit = fit_line(found, cfg.ransac_residual_ft)
        score = (
            sum(1.0 if lb.primary else 0.5 for lb in fit[2]) if fit else 0.0,
            len({lb.token for lb in found}),
        )
        if score > best_score:
            best_k, labels, best_score = k, found, score
    for s in starts:
        if s in probe_gray:
            continue
        nxt = max((lb.token for lb in labels), default=-1) + 1
        labels += _ocr_window(window(s), best_k, upscale, s, nxt, cfg.max_depth_ft)
    log.debug("%s: %d depth hypotheses, orientation k=%d", doc_id, len(labels), best_k)
    return labels


def estimate_scale(img: Image.Image, dc: Box, doc_id: str, cfg: ScaleConfig) -> tuple[DepthScale | None, list[str]]:
    """Fit the depth scale. Returns (scale or None, warnings)."""
    warnings: list[str] = []
    labels = read_depth_labels(img, dc, doc_id, cfg)
    n_tokens = len({lb.token for lb in labels})
    fit = fit_line(labels, cfg.ransac_residual_ft)
    # labels read twice in overlapping windows count once
    uniq = {(round(lb.y / 3), lb.depth): lb for lb in fit[2]} if fit else {}
    if len(uniq) < cfg.min_points:
        warnings.append(f"depth OCR found only {len(uniq)} consistent labels (of {n_tokens} read)")
        return None, warnings
    s, b, inliers = fit
    y = np.array([lb.y for lb in uniq.values()])
    d = np.array([lb.depth for lb in uniq.values()])
    resid = float(np.median(np.abs(d - (s * y + b))))
    confidence = min(1.0, len(uniq) / (cfg.min_points * 2)) * (len(inliers) / max(1, n_tokens)) ** 0.5
    prior = prior_px_per_ft(img, doc_id)
    if prior is not None:
        rel = abs((1.0 / s) / prior - 1.0)
        if rel > cfg.max_slope_disagreement:
            warnings.append(f"OCR scale {1 / s:.2f} px/ft disagrees with filename x DPI {prior:.2f} px/ft ({rel:.0%})")
            confidence *= 0.5
    span = (y.max() - y.min()) / max(1, dc.height)
    if span < 0.3 and dc.height > 3000:
        warnings.append(f"depth labels span only {span:.0%} of the column; extrapolation risk")
        confidence *= 0.7
    scale = DepthScale.linear(
        s, b, "ocr-linear", n_points=len(uniq), residual_ft=resid, confidence=float(round(confidence, 3))
    )
    return scale, warnings
