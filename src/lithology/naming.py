"""Name each group of beds from the legend.

Independent opinions rank the legend swatches per group: a two-way pixel template match, a colour and
stroke-skeleton distance, and optionally frozen ResNet-18 features. Agreement or one decisive opinion names the
group; otherwise best_guess picks a flagged name.
"""

from __future__ import annotations

import re
import warnings
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


def fill_lab(rgb: np.ndarray, ink_luma: int = 110) -> np.ndarray:
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
    """Two-way pixel template match (clusters x swatches): the weaker of piece-in-swatch and swatch-in-piece.

    Matching both ways separates a plain swatch from a variant with extra marks. Flat pieces are skipped, swatches
    whose fill differs by more than ``colour_gate`` (Lab) score 0, and NaN means no usable piece.
    """
    import cv2

    tiled = [np.tile(s, (2, 2, 1)).astype(np.float32) for s in swatches]  # 2x2 so a piece can wrap the pattern
    sw_fill = [fill_lab(s) for s in swatches]

    def best(img: np.ndarray, tpl: np.ndarray) -> float:
        return float(cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED).max())

    out = np.full((len(cluster_patches), len(swatches)), np.nan)
    for c, parts in enumerate(cluster_patches):
        scores: list[np.ndarray] = []
        for p in parts:
            if p.shape[0] < 12 or p.shape[1] < 12:
                continue
            pf = p.astype(np.float32)
            p_tiled = np.tile(pf, (1, 2, 1))  # the swatch window may wrap across the column's width
            fill = fill_lab(p)
            row = np.full(len(swatches), np.nan)
            for k, (s, t, f) in enumerate(zip(swatches, tiled, sw_fill, strict=True)):
                th, tw = min(p.shape[0] - 4, s.shape[0] - 2), min(p.shape[1] - 4, s.shape[1] - 2)
                if th < 8 or tw < 8:
                    continue
                y, x = (p.shape[0] - th) // 2, (p.shape[1] - tw) // 2
                tpl = pf[y : y + th, x : x + tw]
                if tpl.std() < min_detail:
                    continue
                if np.linalg.norm(fill - f) > colour_gate:
                    row[k] = 0.0
                    continue
                sy, sx = (s.shape[0] - th) // 2, (s.shape[1] - tw) // 2
                rev = s[sy : sy + th, sx : sx + tw].astype(np.float32)
                fwd = best(t, tpl)
                back = best(p_tiled, rev) if rev.std() >= min_detail else fwd
                row[k] = min(fwd, back)
            if np.isfinite(row).any():
                scores.append(row)
        if scores:
            with warnings.catch_warnings():  # a swatch no piece could be compared with stays NaN (no opinion)
                warnings.simplefilter("ignore", RuntimeWarning)
                out[c] = np.nanmean(np.stack(scores), axis=0)
    return out


def decide(
    distance: np.ndarray, similarity: np.ndarray | None, cfg: NamingConfig, template: np.ndarray | None = None
) -> Naming:
    """Decide one group.

    ``distance``: classical distance per swatch (lower is closer). ``similarity``: pretrained cosine similarity
    (higher is closer), or None. ``template``: template scores (NaN = no opinion), decisive when near-exact.
    """
    if template is not None and np.isfinite(template).any():
        template = np.nan_to_num(template, nan=-1.0)  # swatches it could not compare have no say
        t, t2, t_margin = _top2(template, higher_is_better=True)
        if template[t] >= cfg.template_min_score:
            if t_margin >= cfg.template_decisive_margin:
                return Naming(t, cfg.confidence_agree, [t, t2], "column is pixel-identical to this legend swatch")
            # pixel-identical to several swatches: the telling marks are missing, so no opinion can settle it
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
    if a == b and (a_sure or b_sure):
        return Naming(a, cfg.confidence_agree, [a, a2 if a2 != a else b2], "classical and pretrained agree")
    if a == b:  # both unsure: the same top pick is a lead, not a decision
        return Naming(None, cfg.confidence_unresolved, list(dict.fromkeys([a, a2, b2])), "both unsure, same lead")
    if a_sure and not b_sure:
        return Naming(a, cfg.confidence_one_sure, [a, b], "classical decisive, pretrained unsure")
    if b_sure and not a_sure:
        return Naming(b, cfg.confidence_one_sure, [b, a], "pretrained decisive, classical unsure")
    cands = list(dict.fromkeys([a, b] + ([b2] if not b_sure else [])))
    why = "both decisive but disagree" if a_sure and b_sure else "neither method can separate the candidates"
    return Naming(None, cfg.confidence_unresolved, cands, why)


def classify_flagged(
    pick: int, candidates: list[int], swatch_fill: np.ndarray, bed_matches_pick: bool, colour_gate: float
) -> tuple[str, list[int]]:
    """Flag a group as 'similar' (2-3 look-alike entries with the bed's fill colour) or 'unresolved'.

    Returns (status, entries).
    """
    cands = list(dict.fromkeys([pick, *candidates]))
    alike = [i for i in cands if np.linalg.norm(swatch_fill[i] - swatch_fill[pick]) <= colour_gate][:3]
    if bed_matches_pick and len(alike) >= 2:
        return "similar", alike
    return "unresolved", cands[:3]


def similar_note(names: list[str]) -> str:
    return "similar types: computer vision could not differentiate " + " or ".join(dict.fromkeys(names))


def unresolved_note(candidates: list[str]) -> str:
    return "computer vision could not differentiate: " + " or ".join(dict.fromkeys(candidates)) + "?"


# ------------------------------------------------------------------------------------------ best guess
# When no opinion is decisive the CSV still gets a flagged name, chosen in three steps:
#   1. rank aggregation over all opinions (the pixel template counts double when it has an opinion);
#   2. variants of one rock ("Clay-shale" / "Sandy clay-shale"): write the plainest entry;
#   3. pixel-identical swatches (b&w "Clay" and "Shale"): write the more common rock.

# A rock word stands for its base rock; "sand" and "sandstone" are the same rock for this purpose.
_SAME_ROCK = {"sand": "sandstone", "silt": "siltstone", "clay": "claystone", "dolostone": "dolomite", "salt": "halite"}


def _rank(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    order = np.argsort(-scores if higher_is_better else scores, kind="stable")
    r = np.empty(len(scores))
    r[order] = np.arange(len(scores))
    return r


def aggregate_rank(
    distance: np.ndarray, similarity: np.ndarray | None, template: np.ndarray | None, strokes: np.ndarray | None = None
) -> np.ndarray:
    """Sum of each opinion's rank per swatch (0 = best); lower total = more likely."""
    total = _rank(distance, higher_is_better=False)
    if strokes is not None and np.isfinite(strokes).all():
        total = total + _rank(strokes, higher_is_better=False)
    if similarity is not None and np.isfinite(similarity).all():
        total = total + _rank(similarity, higher_is_better=True)
    if template is not None and np.isfinite(template).any() and np.nanmax(template) >= 0.5:
        total = total + 2 * _rank(np.nan_to_num(template, nan=-1.0), higher_is_better=True)
    return total


def rock_parts(label: str) -> tuple[frozenset[str], int]:
    """(base rock words, number of other words) of a legend label: "Sandy clay-shale" -> ({claystone, shale}, 1)."""
    from lithology.legend import MODIFIER_WORDS, ROCK_WORDS, correct_word

    nouns = set(ROCK_WORDS) - set(MODIFIER_WORDS) - {"sandy", "silty"}
    words = [correct_word(w).lower() for w in re.findall(r"[A-Za-z]{2,}", label)]
    base = frozenset(_SAME_ROCK.get(w, w) for w in words if w in nouns)
    return base, sum(w not in nouns for w in words)


def _commonness(label: str, common: tuple[str, ...]) -> int:
    base, _ = rock_parts(label)
    ranks = [common.index(w) for w in base if w in common] + [
        common.index(_SAME_ROCK.get(w, w)) for w in base if _SAME_ROCK.get(w, w) in common
    ]
    return min(ranks) if ranks else len(common)


def _tops(
    distance: np.ndarray, similarity: np.ndarray | None, template: np.ndarray | None, strokes: np.ndarray | None
) -> list[int]:
    """Top pick of every opinion that has one (template only when it found a real resemblance)."""
    tops = [int(np.argmin(distance))]
    if similarity is not None and np.isfinite(similarity).all():
        tops.append(int(np.argmax(similarity)))
    if template is not None and np.isfinite(template).any() and np.nanmax(template) >= 0.5:
        tops.append(int(np.nanargmax(template)))
    if strokes is not None and np.isfinite(strokes).all():
        tops.append(int(np.argmin(strokes)))
    return tops


def best_guess(
    candidates: list[int],
    labels: list[str],
    distance: np.ndarray,
    similarity: np.ndarray | None,
    template: np.ndarray | None,
    identical: list[set[int]],
    cfg: NamingConfig,
    strokes: np.ndarray | None = None,
    colour_ok: np.ndarray | None = None,
) -> tuple[int, float, str, bool]:
    """(swatch index, confidence, reason, still flagged) for a group no single opinion settled.

    When any swatch matches the group's fill colour (``colour_ok``), only those can be the guess.
    """
    agg = aggregate_rank(distance, similarity, template, strokes)
    pool = set(candidates)
    if colour_ok is not None and colour_ok.any():
        ok = {i for i in pool if colour_ok[i]}
        # no candidate has the right colour: take the best-ranked swatches that do
        pool = ok or set(sorted(np.flatnonzero(colour_ok).tolist(), key=lambda i: agg[i])[:3])
    tops = [t for t in _tops(distance, similarity, template, strokes) if t in pool or colour_ok is None]
    if len(tops) >= 2 and len(set(tops)) == 1:  # every opinion leans the same way
        if len(tops) >= 3:
            return tops[0], cfg.confidence_one_sure, f"all {len(tops)} opinions lean to this entry", False
        return tops[0], cfg.confidence_lookalike, "both opinions lean to this entry", True
    ranked = sorted(pool, key=lambda i: agg[i])
    top = ranked[0]
    base, _ = rock_parts(labels[top])
    if base:
        same = [i for i in ranked if rock_parts(labels[i])[0] == base]
        if len(same) > 1:  # the opinions split between variants of one rock
            pick = min(same, key=lambda i: (rock_parts(labels[i])[1], agg[i]))
            return pick, cfg.confidence_lookalike, "variants of one rock: wrote the plainest legend entry", True
    twins = next((g for g in identical if top in g), None)
    if twins and len(twins) > 1:
        pick = min(twins, key=lambda i: (rock_parts(labels[i])[1], _commonness(labels[i], cfg.common_rocks), agg[i]))
        why = "legend draws these entries identically: wrote the plainer, more common rock"
        return pick, cfg.confidence_lookalike, why, True
    return top, cfg.confidence_unresolved, "best guess by rank over all opinions", True


def identical_swatches(inner: list[np.ndarray], threshold: float = 0.9) -> list[set[int]]:
    """Groups of legend swatches that are pixel-identical to each other (two-way template match >= threshold)."""
    sim = template_similarity([[s] for s in inner], inner)
    n = len(inner)
    groups: list[set[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(sim[i, j]) and np.isfinite(sim[j, i]) and min(sim[i, j], sim[j, i]) >= threshold:
                g = next((g for g in groups if i in g or j in g), None)
                if g is None:
                    groups.append({i, j})
                else:
                    g |= {i, j}
    return groups
