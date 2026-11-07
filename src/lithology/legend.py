"""Read the lithology legend: find swatches, read their labels, and match interval clusters to them.

Layout-agnostic: labels may sit to the right of their swatch (most mudlogs) or below it (the organisers'
example). The layout is decided per legend by which pairing explains more swatches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

from lithology.config import LegendConfig
from lithology.features import MATCH_SCALE, match_descriptor
from lithology.models import Box, LegendEntry

log = logging.getLogger(__name__)

# Common rock names, used only to correct OCR typos (e.g. "Llmestone" -> "Limestone"); unknown words pass through.
ROCK_WORDS = [
    "anhydrite",
    "argillaceous",
    "basalt",
    "bentonite",
    "breccia",
    "calcareous",
    "calcarenite",
    "chalk",
    "chert",
    "clay",
    "claystone",
    "coal",
    "conglomerate",
    "dolomite",
    "dolomitic",
    "dolostone",
    "evaporite",
    "gypsum",
    "halite",
    "igneous",
    "limestone",
    "marl",
    "marlstone",
    "metamorphic",
    "mudstone",
    "salt",
    "sand",
    "sandstone",
    "shale",
    "silt",
    "siltstone",
    "silty",
    "sandy",
    "tuff",
    "volcanics",
    "granite",
    "gravel",
    "lignite",
    "cement",
    "till",
    "siderite",
    "limonite",
    "pyrite",
    "glauconite",
    "oolitic",
    "fossils",
]


# Legends also list drilling/show symbols and modifier words. They are not rock types, so a column interval is
# never named after them. A label is excluded only when *every* word in it is one of these (so "Silty Sandstone"
# and "Sand and Sandstone" stay rocks, while "Mud loss", "Gas Good" or "Argillaceous" do not).
SYMBOL_WORDS = set(
    [
        "mud",
        "loss",
        "gain",
        "lost",
        "circulation",
        "gas",
        "oil",
        "and",
        "show",
        "survey",
        "deviation",
        "casing",
        "shoe",
        "fit",
        "lot",
        "test",
        "dst",
        "core",
        "sidewall",
        "dry",
        "fresh",
        "salted",
        "water",
        "bit",
        "change",
        "short",
        "trip",
        "marker",
        "connection",
        "background",
        "formation",
        "drilling",
        "symbol",
        "kick",
        "flow",
        "good",
        "fair",
        "poor",
        "cut",
        "fluorescence",
    ]
)
MODIFIER_WORDS = set(
    [
        "sandy",
        "silty",
        "argillaceous",
        "calcareous",
        "dolomitic",
        "carbonaceous",
        "siliceous",
        "marly",
        "glauconite",
        "glauconitic",
        "kaolinite",
        "kaolinitic",
        "pyrite",
        "pyritic",
        "anhydritic",
        "fossil",
        "fossils",
        "fossiliferous",
        "oolitic",
        "micaceous",
        "cherty",
        "ferruginous",
        "tuffaceous",
        "lignitic",
    ]
)


def _fragmentary(text: str) -> bool:
    """True when no word of the text is a known rock, symbol or modifier word (e.g. "rite" from "Anhydrite")."""
    known = set(ROCK_WORDS) | SYMBOL_WORDS | MODIFIER_WORDS
    return not any(correct_word(w).lower() in known for w in re.findall(r"[A-Za-z]{2,}", text))


def is_rock_type(label: str) -> bool:
    """A usable rock name: not only symbol/modifier words, and not a short unknown OCR fragment ("rite", "fio")."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", label)]
    if not words or all(w in SYMBOL_WORDS | MODIFIER_WORDS for w in words):
        return False
    return any(w in ROCK_WORDS for w in words) or max(len(w) for w in words) >= 5


@dataclass
class _Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Phrase:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float


_CONFUSABLE = ("il1tfj|", "o0ce", "s5", "b6h", "rn", "uv", "z2", "g9q")


def _confusable(a: str, b: str) -> bool:
    return any(a in group and b in group for group in _CONFUSABLE)


def correct_word(word: str) -> str:
    """Fix OCR letter substitutions in a purely alphabetic word ("Llmestone" -> "Limestone").

    Only same-length matches with most letters already equal are accepted: OCR swaps look-alike letters, it does
    not add or drop them, so "Dolomites" or "Clay-S" are left exactly as printed.
    """
    low = word.lower()
    if len(low) < 4 or not low.isalpha() or low in ROCK_WORDS:
        return word
    best = None
    for cand in ROCK_WORDS:
        if len(cand) != len(low):
            continue
        diffs = [(a, b) for a, b in zip(low, cand, strict=True) if a != b]
        ok = 0 < len(diffs) <= 2 and all(_confusable(a, b) for a, b in diffs)
        if ok and (best is None or len(diffs) < best[0]):
            best = (len(diffs), cand)
    if best is None:
        return word
    fixed = best[1]
    return fixed.upper() if word.isupper() else (fixed.capitalize() if word[:1].isupper() else fixed)


def clean_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9 ()/&,.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -.,")
    # correct alphabetic runs only, keeping suffixes like "-SCS" and punctuation intact
    return re.sub(r"[A-Za-z]+", lambda m: correct_word(m.group(0)), text)


def _ocr_words(gray: np.ndarray, upscale: float, min_conf: float) -> list[_Word]:
    pil = Image.fromarray(gray)
    if upscale != 1.0:
        pil = pil.resize((round(pil.width * upscale), round(pil.height * upscale)), Image.Resampling.LANCZOS)
    d = pytesseract.image_to_data(pil, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"], strict=False):
        t = _strip_glued(t.strip())
        if not t or float(c) < min_conf or not re.search(r"[A-Za-z]{2,}", t):
            continue
        s = 1.0 / upscale
        words.append(_Word(t, int(x * s), int(y * s), int((x + w) * s), int((y + h) * s), float(c)))
    return words


def _strip_glued(text: str) -> str:
    """Remove swatch-border ink read as characters at a word's ends ("|Sandstone", "4Siltstone", "Marly_")."""
    core = re.sub(r"^[^A-Za-z(]+(?=[A-Za-z]{3})", "", text)
    return re.sub(r"(?<=[A-Za-z)])[^A-Za-z)]+$", "", core) if re.search(r"[A-Za-z]{3}", core) else core


def _merge_words(primary: list[_Word], extra: list[_Word]) -> list[_Word]:
    """Primary OCR pass plus words from a second pass that fall where the primary pass read nothing."""

    def overlaps(a: _Word, b: _Word) -> bool:
        return min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0)

    return primary + [w for w in extra if not any(overlaps(w, p) for p in primary)]


def _phrases(words: list[_Word]) -> list[_Phrase]:
    """Join words on the same line that are close together into phrases."""
    words = sorted(words, key=lambda w: (round(w.cy / max(4, w.h)), w.x0))
    lines: list[list[_Word]] = []
    for w in sorted(words, key=lambda w: (w.y0, w.x0)):
        for line in lines:
            last = line[-1]
            if abs(w.cy - last.cy) < 0.6 * max(w.h, last.h) and 0 <= w.x0 - last.x1 < 1.2 * max(w.h, last.h):
                line.append(w)
                break
        else:
            lines.append([w])
    out = []
    for line in lines:
        out.append(
            _Phrase(
                " ".join(w.text for w in line),
                min(w.x0 for w in line),
                min(w.y0 for w in line),
                max(w.x1 for w in line),
                max(w.y1 for w in line),
                float(np.mean([w.conf for w in line])),
            )
        )
    return out


def _is_wordlike(w: _Word) -> bool:
    """Confident, word-shaped text; a known rock name is trusted at lower confidence."""
    letters = re.sub(r"[^A-Za-z]", "", w.text)
    if len(letters) < 3 or len(letters) < 0.6 * len(w.text):
        return False
    known = any(correct_word(part).lower() in ROCK_WORDS for part in re.findall(r"[A-Za-z]+", w.text))
    return w.conf >= (30 if known else 60)


def _swatch_candidates(gray: np.ndarray, words: list[_Word], min_px: int) -> list[Box]:
    """Graphic blobs that are not text.

    Long ruled lines (frames, separators) are removed first so swatches touching them stay separate. A small
    closing joins a pattern's dots and dashes into one component. A component is text, not a swatch, when
    confident word-like OCR boxes cover most of it. Boxes nested inside a larger candidate are dropped.
    """
    H, W = gray.shape
    ink = (gray < 200).astype(np.uint8)
    # Rules are often anti-aliased grey, so find them with a lighter threshold than the ink itself.
    faint = (gray < 235).astype(np.uint8)
    hl = cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 4), 1)))
    vl = cv2.morphologyEx(faint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, H // 3))))
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    ink[lines > 0] = 0
    real = [w for w in words if _is_wordlike(w)]
    # Cut confident words out, so a label printed touching its swatch does not fuse with it. Only word-like OCR
    # is cut: "text" read inside a pattern is low-confidence or not word-shaped, so swatches stay intact.
    for w in real:
        ink[w.y0 : w.y1, w.x0 : w.x1] = 0
    th = int(np.median([w.h for w in real])) if real else 12
    k = max(3, th // 3)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 2))))
    n, _, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    text_mask = np.zeros((H, W), bool)
    for w in real:
        text_mask[w.y0 : w.y1, w.x0 : w.x1] = True
    boxes = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if w < min_px or h < min_px // 2 or w > 0.45 * W or h > 0.45 * H:
            continue
        if w / max(h, 1) > 15 or h / max(w, 1) > 6:
            continue
        if text_mask[y : y + h, x : x + w].mean() > 0.5:
            continue
        boxes.append(Box(int(x), int(y), int(x + w), int(y + h)))

    def inside(a: Box, b: Box) -> bool:
        return a is not b and b.x0 <= a.x0 and b.y0 <= a.y0 and a.x1 <= b.x1 and a.y1 <= b.y1

    return [a for a in boxes if not any(inside(a, b) for b in boxes)]


def _split_glued(swatches: list[Box], gray: np.ndarray) -> tuple[list[Box], set[int]]:
    """Trim swatches fused with an unread label printed against them.

    Swatches in one legend share a size. A candidate as tall as the typical swatch but far wider, whose extra part
    looks different from its swatch part (text on white, not more of the same pattern), is a swatch plus glued
    text: keep its left part at the typical width (the text is then re-read). Returns (boxes, trimmed indices).
    """
    if len(swatches) < 4:
        return swatches, set()
    mw = float(np.median([b.width for b in swatches]))
    mh = float(np.median([b.height for b in swatches]))
    out, trimmed = [], set()
    for i, b in enumerate(swatches):
        cut = b.x0 + int(round(mw))
        if b.width > 1.6 * mw and abs(b.height - mh) <= 0.2 * mh:
            left = gray[b.y0 : b.y1, b.x0 : cut].astype(float)
            right = gray[b.y0 : b.y1, cut : b.x1].astype(float)
            if right.mean() - left.mean() > 40:  # extra part much lighter: text on paper, not more pattern
                out.append(Box(b.x0, b.y0, cut, b.y1))
                trimmed.add(i)
                continue
        out.append(b)
    return out, trimmed


def _assign(swatches: list[Box], phrases: list[_Phrase], layout: str, th: float, max_gap: float) -> list[list[int]]:
    """Give every phrase to at most one swatch: the nearest one on its left ('right' layout) or above it
    ('below' layout). Returns, per swatch, the indices of its phrases in reading order, keeping only phrases that
    chain on from the swatch (a phrase further away than ``max_gap``, or detached from the previous one, is
    not part of this label)."""
    owner: dict[int, int] = {}
    for j, p in enumerate(phrases):
        pcx, pcy = (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2
        best = None
        for i, s in enumerate(swatches):
            if layout == "right":
                gap = p.x0 - s.x1
                aligned = s.y0 - 0.3 * th <= pcy <= s.y1 + 0.3 * th
                off = abs(pcy - (s.y0 + s.y1) / 2)
            else:
                gap = p.y0 - s.y1
                aligned = s.x0 - th <= pcx <= s.x1 + th
                off = abs(pcx - (s.x0 + s.x1) / 2)
            if aligned and -2 <= gap <= max_gap + 4 * th and (best is None or (gap, off) < best[0]):
                best = ((gap, off), i)
        if best is not None:
            owner[j] = best[1]
    per: list[list[int]] = [[] for _ in swatches]
    for j, i in owner.items():
        per[i].append(j)
    out = []
    for i, js in enumerate(per):
        s = swatches[i]
        if layout == "right":
            js.sort(key=lambda j: phrases[j].x0)
            chain, edge = [], s.x1
            for j in js:
                if phrases[j].x0 - edge > (max_gap if not chain else 2.5 * th):
                    break
                chain.append(j)
                edge = phrases[j].x1
        else:
            js.sort(key=lambda j: (phrases[j].y0, phrases[j].x0))
            chain, edge = [], s.y1
            for j in js:
                if phrases[j].y0 - edge > (max_gap if not chain else 1.0 * th):
                    break
                chain.append(j)
                edge = max(edge, phrases[j].y1)
            chain.sort(key=lambda j: (round(phrases[j].y0 / max(th, 1)), phrases[j].x0))
        out.append(chain)
    return out


def _reread_label(
    gray: np.ndarray, s: Box, swatches: list[Box], layout: str, th: float, upscale: float, min_conf: float
) -> tuple[str, float]:
    """OCR the strip beside (or under) a swatch as a single text line (Tesseract --psm 7)."""
    H, W = gray.shape
    if layout == "right":
        nxt = min((o.x0 for o in swatches if o is not s and o.x0 > s.x1 and o.y0 < s.y1 and o.y1 > s.y0), default=W)
        x0, x1 = s.x1 + 2, min(W, nxt - 2, s.x1 + int(30 * th))
        y0, y1 = max(0, s.y0 - 2), min(H, s.y1 + 2)
    else:
        x0, x1 = max(0, s.x0 - int(th)), min(W, s.x1 + int(th))
        y0, y1 = s.y1 + 2, min(H, s.y1 + int(3.5 * th))
    if x1 - x0 < 8 or y1 - y0 < 6:
        return "", 0.0
    pil = Image.fromarray(np.ascontiguousarray(gray[y0:y1, x0:x1]))
    pil = pil.resize((round(pil.width * upscale * 1.5), round(pil.height * upscale * 1.5)), Image.Resampling.LANCZOS)
    psm = "7" if layout == "right" else "6"
    d = pytesseract.image_to_data(pil, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    toks = [
        (t.strip(), float(c))
        for t, c in zip(d["text"], d["conf"], strict=False)
        if t.strip() and float(c) >= min_conf and re.search(r"[A-Za-z]", t)
    ]
    if not toks:
        return "", 0.0
    return " ".join(t for t, _ in toks), float(np.mean([c for _, c in toks]))


def read_legend(rgb: np.ndarray, origin: tuple[int, int], dpi: float, cfg: LegendConfig) -> list[LegendEntry]:
    """Swatch + label pairs from a legend crop. Swatch boxes are returned in full-image pixels."""
    gray = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    upscale = 2.0 if dpi <= 200 else 1.0
    words = _ocr_words(gray, upscale, cfg.ocr_min_conf)
    if dpi <= 250:  # small text: a second zoom level reads labels the first one misses
        words = _merge_words(words, _ocr_words(gray, 3.0 - upscale, cfg.ocr_min_conf))
    swatches, glued = _split_glued(_swatch_candidates(gray, words, cfg.min_swatch_px), gray)

    def in_swatch(w: _Word) -> bool:
        cx, cy = (w.x0 + w.x1) / 2, w.cy
        return any(s.x0 <= cx <= s.x1 and s.y0 <= cy <= s.y1 for s in swatches)

    words = [w for w in words if not in_swatch(w)]  # "text" read inside a pattern is not a label
    phrases = _phrases(words)
    th = float(np.median([p.y1 - p.y0 for p in phrases])) if phrases else 12.0
    max_gap = cfg.max_swatch_gap_frac * gray.shape[1] + th
    found = {layout: _assign(swatches, phrases, layout, th, max_gap) for layout in ("right", "below")}
    n_right = sum(bool(c) for c in found["right"])
    n_below = sum(bool(c) for c in found["below"])
    layout = "right" if n_right >= n_below else "below"
    entries = []
    for s, chain in sorted(zip(swatches, found[layout], strict=True), key=lambda sc: (sc[0].y0, sc[0].x0)):
        if chain:
            text = " ".join(phrases[j].text for j in chain)
            conf = float(np.mean([phrases[j].conf for j in chain]))
            # labels under swatches wrap onto two lines, and labels glued to a swatch lose their first letters:
            # in both cases page OCR can miss part of the label, so re-read its strip and keep the fuller reading
            if layout == "below" or (swatches.index(s) in glued and _fragmentary(text)):
                rr, rr_conf = _reread_label(gray, s, swatches, layout, th, upscale, cfg.ocr_min_conf)
                first = clean_label(text).split()[:1]
                if (
                    first
                    and len(re.sub("[^A-Za-z]", "", rr)) > len(re.sub("[^A-Za-z]", "", text))
                    and first[0].lower() in clean_label(rr).lower()
                ):
                    text, conf = rr, rr_conf
        else:  # page-level OCR missed this label: re-read just the strip where it should be
            text, conf = _reread_label(gray, s, swatches, layout, th, upscale, cfg.ocr_min_conf)
            # A fallback read must name a known rock, so specks and stray glyphs do not become entries.
            if not any(w.lower() in ROCK_WORDS for w in re.findall(r"[A-Za-z]+", clean_label(text))):
                continue
        label = clean_label(text)
        if len(re.sub(r"[^A-Za-z]", "", label)) < cfg.min_label_chars:
            continue
        entries.append(
            LegendEntry(
                label=label,
                swatch=Box(s.x0 + origin[0], s.y0 + origin[1], s.x1 + origin[0], s.y1 + origin[1]),
                ocr_confidence=conf,
                is_rock=is_rock_type(label),
            )
        )
    log.debug(
        "legend: %d swatches, %d phrases, layout=%s, %d entries",
        len(swatches),
        len(phrases),
        layout,
        len(entries),
    )
    return entries


def swatch_descriptor(rgb_full_swatch: np.ndarray, ink_luma: int) -> np.ndarray:
    """Match descriptor of a swatch's interior (shrunk 15% per side so its border does not count)."""
    h, w = rgb_full_swatch.shape[:2]
    my, mx = max(1, int(h * 0.15)), max(1, int(w * 0.15))
    inner = rgb_full_swatch[my : h - my, mx : w - mx] if h > 2 * my + 2 and w > 2 * mx + 2 else rgb_full_swatch
    return match_descriptor(inner, ink_luma)


@dataclass(frozen=True)
class Match:
    swatch: int  # index into the legend entries
    distance: float
    margin: float  # distance to the runner-up minus distance to the winner (bigger = more certain)


def match_distances(cluster_desc: np.ndarray, swatch_desc: np.ndarray) -> np.ndarray:
    """Fixed-scale distance (clusters x swatches) between match descriptors; lower = more alike."""
    a = cluster_desc / MATCH_SCALE
    b = swatch_desc / MATCH_SCALE
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def match_clusters(cluster_desc: np.ndarray, swatch_desc: np.ndarray) -> list[Match]:
    """Nearest swatch per cluster on fixed-scale match descriptors."""
    a = cluster_desc / MATCH_SCALE
    b = swatch_desc / MATCH_SCALE
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    out = []
    for row in dist:
        order = np.argsort(row)
        second = float(row[order[1]]) if len(order) > 1 else float(row[order[0]]) + 10.0
        out.append(Match(int(order[0]), float(row[order[0]]), second - float(row[order[0]])))
    return out
