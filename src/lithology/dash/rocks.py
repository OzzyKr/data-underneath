"""Rock names across logs: a display name, a rock family for cross-well comparison, and a calm colour.

Colours follow each log's own legend (the user's choice), toned down so the dashboard reads as an engineering tool:
a chromatic swatch keeps its hue at a fixed, muted saturation and lightness; a black-and-white swatch has no colour,
so the rock takes its family's colour. Families use the same hue conventions the legends use (sand yellow, clay
green, limestone blue, dolomite purple, salt pink).
"""

from __future__ import annotations

import colorsys
import re

UNRESOLVED = "Unresolved"

FAMILY_COLOURS: dict[str, str] = {
    "Sand": "#DCC577",
    "Silt": "#C2A578",
    "Shale": "#95A4B3",
    "Clay": "#84BA8E",
    "Limestone": "#74A9D6",
    "Dolomite": "#B596D4",
    "Salt": "#DB9DB1",
    "Coal": "#6A717C",
    "Volcanic": "#D29372",
    "Conglomerate": "#C9A27E",
    "Other": "#7E8794",
    UNRESOLVED: "#5E6670",
}
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


def muted(rgb: tuple[int, int, int]) -> str:
    """A legend colour at dashboard strength: same hue, saturation <= 0.42, lightness 0.60-0.68."""
    h, lightness, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
    s = min(s, 0.42)
    lightness = min(max(lightness, 0.60), 0.68)
    return _hex(*colorsys.hls_to_rgb(h, lightness, s))


def shade(hex_colour: str, step: int) -> str:
    """Tell apart rocks that share a colour (Clay-S and Clay-M share the legend's green): alternate darker/lighter."""
    if step == 0:
        return hex_colour
    h, lightness, s = colorsys.rgb_to_hls(*_from_hex(hex_colour))
    delta = [0, -0.14, 0.1, -0.24, 0.16][step % 5]
    return _hex(*colorsys.hls_to_rgb(h, min(max(lightness + delta, 0.28), 0.84), s))


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
        base = muted(rgb) if rgb is not None and chroma >= CHROMA_MIN else FAMILY_COLOURS[rock_family(name)]
        n = used.get(base, 0)
        used[base] = n + 1
        out[name] = shade(base, n)
    return out
