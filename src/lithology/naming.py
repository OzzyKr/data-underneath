"""Name each group of intervals from the legend, or say honestly that it cannot be told apart.

Two independent opinions rank the legend swatches for each group:
  classical  - fixed-scale colour + stroke-skeleton descriptor distance (always on)
  pretrained - frozen ResNet-18 feature similarity (tier 2, optional)

Decision rule (explainable, no learned weights):
  both agree on the best swatch                        -> named, high confidence
  they disagree, exactly one has a decisive margin     -> that one's choice, medium confidence
  otherwise (both unsure, or both sure but different)  -> unresolved: "computer vision could not differentiate"
With only the classical opinion, a group is named when its margin is decisive and unresolved otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lithology.config import NamingConfig


@dataclass
class Naming:
    swatch: int | None  # index of the chosen legend entry; None = unresolved
    confidence: float
    candidates: list[int] = field(default_factory=list)  # most likely swatches, best first
    reason: str = ""


def _top2(scores: np.ndarray, higher_is_better: bool) -> tuple[int, int, float]:
    order = np.argsort(-scores if higher_is_better else scores)
    best = int(order[0])
    if len(order) == 1:
        return best, best, float("inf")
    second = int(order[1])
    return best, second, float(abs(scores[best] - scores[second]))


def _fill_lab(rgb: np.ndarray, ink_luma: int = 110) -> np.ndarray:
    """Lab colour of the background (non-ink) pixels."""
    from skimage.color import rgb2lab

    px = rgb.reshape(-1, 3).astype(np.float64)
    luma = px @ np.array([0.299, 0.587, 0.114])
    fill = px[luma >= ink_luma] if (luma >= ink_luma).any() else px
    return rgb2lab(np.clip(fill.mean(0) / 255.0, 0, 1)[None, None])[0, 0]


def template_similarity(
    cluster_patches: list[list[np.ndarray]],
    swatches: list[np.ndarray],
    min_detail: float = 12.0,
    colour_gate: float = 12.0,
) -> np.ndarray:
    """Pixel template match (clusters x swatches): best normalised correlation of column pieces inside each swatch.

    When a log draws its column and legend with the same software, a column piece is pixel-identical to its
    swatch (score ~1.0). Pieces without real pattern detail (flat colour, std < ``min_detail``) say nothing and
    are skipped, so thin flat slivers cannot produce false perfect scores. Correlation ignores average colour,
    so a swatch whose background colour differs by more than ``colour_gate`` (Lab) scores 0. NaN = no usable piece.
    """
    import cv2

    tiled = [np.tile(s, (2, 2, 1)).astype(np.float32) for s in swatches]  # 2x2 so a piece can wrap the pattern
    sw_fill = [_fill_lab(s) for s in swatches]
    out = np.full((len(cluster_patches), len(swatches)), np.nan)
    for c, parts in enumerate(cluster_patches):
        scores: list[np.ndarray] = []
        for p in parts:
            th = min(p.shape[0] - 4, min(s.shape[0] for s in swatches) - 2)
            tw = min(p.shape[1] - 4, min(s.shape[1] for s in swatches) - 2)
            if th < 8 or tw < 8:
                continue
            y, x = (p.shape[0] - th) // 2, (p.shape[1] - tw) // 2
            tpl = p[y : y + th, x : x + tw].astype(np.float32)
            if tpl.std() < min_detail:
                continue
            fill = _fill_lab(p)
            scores.append(
                np.array(
                    [
                        float(cv2.matchTemplate(t, tpl, cv2.TM_CCOEFF_NORMED).max())
                        if np.linalg.norm(fill - f) <= colour_gate
                        else 0.0
                        for t, f in zip(tiled, sw_fill, strict=True)
                    ]
                )
            )
        if scores:
            out[c] = np.mean(scores, axis=0)
    return out


def decide(
    distance: np.ndarray, similarity: np.ndarray | None, cfg: NamingConfig, template: np.ndarray | None = None
) -> Naming:
    """Decide one group. ``distance``: classical distance per swatch (lower = closer).
    ``similarity``: pretrained cosine similarity per swatch (higher = closer), or None when tier 2 is off.
    ``template``: pixel template scores per swatch (NaN = no opinion); a near-exact match is decisive on its own."""
    if template is not None and np.isfinite(template).all():
        t, t2, t_margin = _top2(template, higher_is_better=True)
        if template[t] >= cfg.template_min_score:
            if t_margin >= cfg.template_decisive_margin:
                return Naming(t, cfg.confidence_agree, [t, t2], "column is pixel-identical to this legend swatch")
            # Pixel-identical to several swatches: the marks that tell them apart are not in the column, so the
            # statistical opinions (looking at the same pixels) cannot settle it either.
            tied = [int(i) for i in np.argsort(-template) if template[i] >= template[t] - cfg.template_decisive_margin]
            return Naming(None, cfg.confidence_unresolved, tied, "column matches several swatches equally")
    a, a2, a_margin = _top2(distance, higher_is_better=False)
    a_sure = a_margin >= cfg.classical_decisive_margin
    if similarity is None:
        if a_sure:
            return Naming(a, cfg.confidence_one_sure, [a, a2], "classical match is decisive")
        return Naming(None, cfg.confidence_unresolved, [a, a2], "classical match too close to call")
    b, b2, b_margin = _top2(similarity, higher_is_better=True)
    b_sure = b_margin >= cfg.pretrained_decisive_margin
    if a == b:
        return Naming(a, cfg.confidence_agree, [a, a2 if a2 != a else b2], "classical and pretrained agree")
    if a_sure and not b_sure:
        return Naming(a, cfg.confidence_one_sure, [a, b], "classical decisive, pretrained unsure")
    if b_sure and not a_sure:
        return Naming(b, cfg.confidence_one_sure, [b, a], "pretrained decisive, classical unsure")
    cands = list(dict.fromkeys([a, b] + ([b2] if not b_sure else [])))
    why = "both decisive but disagree" if a_sure and b_sure else "neither method can separate the candidates"
    return Naming(None, cfg.confidence_unresolved, cands, why)


def unresolved_note(candidates: list[str]) -> str:
    return "computer vision could not differentiate: " + " or ".join(dict.fromkeys(candidates)) + "?"
