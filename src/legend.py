"""Parse a lithology legend: find swatches, read their names, describe them with the same
features as the column so clusters can be matched to rock names."""
import difflib
import re
import cv2
import numpy as np
import pytesseract
from src.segment import features, ink_mask

VOCAB = """limestone dolomite dolomitic anhydrite halite salt coal clay claystone shale shaly
siltstone silty sandstone sand sandy and conglomerate breccia chert chalk marl marlstone gypsum
bentonite cement granite igneous metamorphic tuff welded till siderite limonite mudstone calcareous
argillaceous lignite pepper choke colored gray grey unknown no sample volcanic basalt""".split()


def _fix_word(w):
    """Correct OCR slips ('Slity' -> 'Silty') against a small lithology vocabulary."""
    core = re.sub(r"[^A-Za-z]", "", w)
    if not core:
        return ""
    m = difflib.get_close_matches(core.lower(), VOCAB, n=1, cutoff=0.75)
    if not m:
        return core
    fixed = m[0]
    return fixed.upper() if core.isupper() else (fixed.capitalize() if core[0].isupper() else fixed)


def _plausible(word, conf):
    """Keep lithology words (fuzzy vocabulary match) or very confident clean words; drop pattern noise."""
    core = re.sub(r"[^A-Za-z]", "", word)
    if len(core) < 3:
        return False
    if difflib.get_close_matches(core.lower(), VOCAB, n=1, cutoff=0.75):
        return conf >= 40
    return conf >= 90 and len(core) >= 4 and re.search(r"[aeiouyAEIOUY]", core) \
        and not re.search(r"(.)\1\1", core.lower())


def _ocr(img, psm, s):
    im = img if s == 1 else img.resize((img.width * s, img.height * s))
    d = pytesseract.image_to_data(im, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    return [dict(text=t, conf=float(c), x0=l / s, y0=tp / s, x1=(l + w) / s, y1=(tp + h) / s)
            for t, l, tp, w, h, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"])
            if t.strip() and _plausible(t, float(c))]


def _words(img):
    """OCR legend words with two page modes (sparse + block) and upscaling for small text; merge duplicates."""
    first = _ocr(img, 11, 1)
    h = np.median([w["y1"] - w["y0"] for w in first]) if first else 10
    s = int(np.clip(round(28 / max(h, 1)), 1, 4))
    cands = _ocr(img, 11, s) + _ocr(img, 6, s)
    words = []
    for w in sorted(cands, key=lambda w: -w["conf"]):
        if not any(min(w["x1"], o["x1"]) - max(w["x0"], o["x0"]) > 0.3 * (w["x1"] - w["x0"]) and
                   min(w["y1"], o["y1"]) - max(w["y0"], o["y0"]) > 0.3 * (w["y1"] - w["y0"]) for o in words):
            words.append(w)
    for w in words:
        for k in ("x0", "y0", "x1", "y1"):
            w[k] = int(round(w[k]))
    return words


def _lines(words):
    """Group words into text lines (same baseline, small horizontal gap)."""
    lines = []
    for w in sorted(words, key=lambda w: (w["y0"], w["x0"])):
        h = w["y1"] - w["y0"]
        for ln in lines:
            cy = (ln["y0"] + ln["y1"]) / 2
            if abs((w["y0"] + w["y1"]) / 2 - cy) < 0.5 * h and 0 <= w["x0"] - ln["x1"] < 1.5 * h:
                ln["words"].append(w["text"])
                ln["x1"], ln["y0"], ln["y1"] = w["x1"], min(ln["y0"], w["y0"]), max(ln["y1"], w["y1"])
                break
        else:
            lines.append(dict(words=[w["text"]], **{k: w[k] for k in ("x0", "y0", "x1", "y1")}))
    return lines


def _blobs(a, words, h):
    """Candidate swatch boxes: connected non-background regions, text erased."""
    hsv_sat = (a.max(2).astype(int) - a.min(2)) / 255
    mask = (ink_mask(a) | (hsv_sat > 0.12) | (a.mean(2) < 225)).astype(np.uint8)
    for w in words:                                   # erase label text
        mask[max(0, w["y0"] - 2):w["y1"] + 2, max(0, w["x0"] - 2):w["x1"] + 2] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    big = [i for i in range(1, n) if st[i, 2] > 0.6 * a.shape[1] or st[i, 3] > 0.6 * a.shape[0]]
    if big:
        mask[np.isin(lab, big)] = 0                   # legend frame / long rules
    kx = max(3, int(2 * h))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [(x, y, x + bw, y + bh) for x, y, bw, bh, area in stats[1:]
             if bw >= 0.8 * h and bh >= 0.6 * h and area >= 0.5 * h * h and bw < 0.6 * a.shape[1]]
    # drop boxes contained in a bigger one (pattern pieces inside a framed swatch)
    keep = []
    for b in boxes:
        if not any(o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3] for o in boxes):
            keep.append(b)
    return keep


def _pair(line, blobs, h, layout):
    """Swatch for a label line: directly left of it ('right' layout) or directly above it ('below')."""
    best, bd = None, 1e9
    for b in blobs:
        x0, y0, x1, y1 = b
        if layout == "right":
            yov = min(y1, line["y1"]) - max(y0, line["y0"])
            if yov > 0.4 * (line["y1"] - line["y0"]) and x1 <= line["x0"] + 2:
                d = line["x0"] - x1
                if d < 3 * h and d < bd:
                    best, bd = b, d
        else:
            cx = (line["x0"] + line["x1"]) / 2
            if x0 <= cx <= x1 and y1 <= line["y0"] + 2:
                d = line["y0"] - y1
                if d < 3.5 * h and d < bd:
                    best, bd = b, d
    return best


def parse_legend(img):
    """Returns [{'name', 'box', 'feat'}] for every named swatch."""
    a = np.array(img.convert("RGB"))
    words = _words(img)
    if not words:
        return []
    h = float(np.median([w["y1"] - w["y0"] for w in words]))
    lines = _lines(words)
    blobs = _blobs(a, words, h)
    n_right = sum(_pair(ln, blobs, h, "right") is not None for ln in lines)
    n_below = sum(_pair(ln, blobs, h, "below") is not None for ln in lines)
    layout = "right" if n_right >= n_below else "below"
    named = {}
    for ln in lines:
        b = _pair(ln, blobs, h, layout)
        if b is not None:
            named.setdefault(b, []).append(ln)
    out = []
    for b, lns in named.items():
        name = " ".join(_fix_word(w) for ln in sorted(lns, key=lambda l: l["y0"]) for w in ln["words"])
        name = re.sub(r"\s+", " ", name).strip()
        x0, y0, x1, y1 = b
        px, py = max(1, (x1 - x0) // 7), max(1, (y1 - y0) // 7)         # stay inside any frame
        patch = a[y0 + py:y1 - py, x0 + px:x1 - px]
        if patch.shape[0] < 4 or patch.shape[1] < 4 or not name:
            continue
        hp = patch.shape[0]
        _, f, _ = features(patch, win=hp, step=hp, core=hp)
        out.append(dict(name=name, box=b, feat=f[0]))
    return out


if __name__ == "__main__":
    import sys
    from src.data import load_crops
    for n in sys.argv[1:]:
        ann, c = load_crops(n)
        sw = parse_legend(c["lithology_legend"])
        print(f"== {n[:40]}: {len(sw)} swatches")
        for s in sorted(sw, key=lambda s: (s["box"][1], s["box"][0])):
            print("  ", s["box"], s["name"])
