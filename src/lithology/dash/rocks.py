"""Rock names across logs: a display name, a rock family for cross-well comparison, and a colour.

Colours follow each log's own legend (the user's choice): a chromatic swatch is snapped to the nearest hue of a
validated categorical palette (colour-blind separation, contrast and lightness checked on the dashboard's dark
surface with the dataviz validator), so a legend's yellow sand stays yellow and its green clay stays green, at a
strength that reads on screen. A black-and-white swatch has no colour, so the rock takes its family's colour.
Families use the same hue conventions the legends use (sand yellow, clay green, limestone blue, dolomite violet,
salt magenta).
"""

from __future__ import annotations

import colorsys
import re

UNRESOLVED = "Unresolved"

# Validated dark-mode categorical slots, mapped to rock families by convention. Chart stacking order below keeps
# neighbouring segments in the validated adjacent order (all checks pass on #121419 and #171A20).
FAMILY_COLOURS: dict[str, str] = {
    "Limestone": "#3987E5",
    "Silt": "#D95926",
    "Shale": "#199E70",
    "Sand": "#C98500",
    "Salt": "#D55181",
    "Clay": "#008300",
    "Dolomite": "#9085E9",
    "Volcanic": "#E66767",
    "Coal": "#7E8794",
    "Conglomerate": "#7E8794",
    "Other": "#7E8794",
    UNRESOLVED: "#4A515C",
}
CHART_FAMILIES = ["Limestone", "Silt", "Shale", "Sand", "Salt", "Clay", "Dolomite", "Volcanic", "Other", UNRESOLVED]
FOLD_TO_OTHER = {"Coal", "Conglomerate"}  # rare in this data set; folded so charts keep <= 8 hues
SLOTS = ["#3987E5", "#D95926", "#199E70", "#C98500", "#D55181", "#008300", "#9085E9", "#E66767"]
FAMILY_ORDER = list(FAMILY_COLOURS)

# keyword -> family. The family of a name is the keyword that appears LAST in it: English rock names put the head
# noun last ("Sandy Clay-Shale" is a shale, "Dolomitic limestone" a limestone, "Silty Sandstone" a sandstone).
_KEYWORDS: list[tuple[str, str]] = [
    ("sand", "Sand"), ("sst", "Sand"), ("grit", "Sand"),
    ("silt", "Silt"), ("slt", "Silt"),
    ("shale", "Shale"), ("shaly", "Shale"),
    ("clay", "Clay"), ("mud", "Clay"), ("argill", "Clay"),
    ("lime", "Limestone"), ("chalk", "Limestone"), ("marl", "Limestone"), ("calc", "Limestone"),
    ("carbonate", "Limestone"), ("oolit", "Limestone"),
    ("dolo", "Dolomite"),
    ("salt", "Salt"), ("halite", "Salt"), ("anhyd", "Salt"), ("gyps", "Salt"), ("evap", "Salt"),
    ("coal", "Coal"), ("lignite", "Coal"),
    ("tuff", "Volcanic"), ("volcan", "Volcanic"), ("basalt", "Volcanic"), ("igneous", "Volcanic"),
    ("congl", "Conglomerate"), ("breccia", "Conglomerate"), ("gravel", "Conglomerate"),
]  # fmt: skip
# words that modify a rock rather than name it ("sandy", "silty", "calcareous", "dolomitic")
_MODIFIER = re.compile(r"(sandy|silty|shaly|calcareous|dolomitic|argillaceous|limy|clayey)$")


def display_name(label: str) -> str:
    """'SHALE' -> 'Shale'; mixed-case labels and codes ('Sand-SIS', 'Clay-S') are kept as the legend wrote them."""
    label = " ".join(label.split())
    if label.isupper() and len(label) > 3:
        return " ".join(w if any(c.isdigit() for c in w) else w.capitalize() for w in label.split(" "))
    return label


def rock_family(label: str) -> str:
    low = label.lower()
    if low == UNRESOLVED.lower():
        return UNRESOLVED
    best: tuple[int, int, str] | None = None  # (is_head_noun, position, family)
    for word_m in re.finditer(r"[a-z]+", low):
        word = word_m.group(0)
        for kw, fam in _KEYWORDS:
            i = word.find(kw)
            if i < 0:
                continue
            head = 0 if _MODIFIER.search(word) else 1
            key = (head, word_m.start() + i, fam)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else "Other"


def _hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(v * 255))) for v in (r, g, b)))


def _from_hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _hue(rgb: tuple[float, float, float]) -> float:
    return colorsys.rgb_to_hls(*rgb)[0]


def snap(rgb: tuple[int, int, int]) -> str:
    """The validated slot whose hue is closest to a legend colour's hue."""
    h = _hue(tuple(c / 255 for c in rgb))  # type: ignore[arg-type]

    def dist(slot: str) -> float:
        d = abs(_hue(_from_hex(slot)) - h)
        return min(d, 1 - d)

    return min(SLOTS, key=dist)


def shade(hex_colour: str, step: int) -> str:
    """Tell apart rocks that share a colour (Clay-S and Clay-M share the legend's green): same hue, lighter first."""
    if step == 0:
        return hex_colour
    h, lightness, s = colorsys.rgb_to_hls(*_from_hex(hex_colour))
    delta = [0, 0.17, 0.32, -0.1, 0.45][step % 5]
    new = lightness + delta
    if new > 0.82:
        new = lightness - delta / 2
    return _hex(*colorsys.hls_to_rgb(h, min(max(new, 0.22), 0.82), s * 0.8))


CHROMA_MIN = 40.0  # below this a swatch is black-and-white (its pattern, not its colour, identifies the rock)


def log_colours(rocks: list[tuple[str, tuple[int, int, int] | None, float]]) -> dict[str, str]:
    """Colour per rock name for one log. ``rocks`` = (name, legend rgb or None, chroma) in order of appearance."""
    out: dict[str, str] = {}
    used: dict[str, int] = {}
    for name, rgb, chroma in rocks:
        if name in out:
            continue
        if rock_family(name) == UNRESOLVED:
            out[name] = FAMILY_COLOURS[UNRESOLVED]
            continue
        base = snap(rgb) if rgb is not None and chroma >= CHROMA_MIN else FAMILY_COLOURS[rock_family(name)]
        n = used.get(base, 0)
        used[base] = n + 1
        out[name] = shade(base, n)
    return out
