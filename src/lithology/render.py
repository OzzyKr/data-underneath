"""Outputs: the results CSV and the annotated image (drawing code ported from the organisers' notebook)."""

from __future__ import annotations

import colorsys
import csv
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lithology.io import crop_rgb
from lithology.models import CSV_COLUMNS, Box, DocumentBoxes, Interval

PALETTE = [
    (40, 160, 60),
    (160, 100, 40),
    (240, 190, 0),
    (150, 60, 170),
    (0, 150, 150),
    (230, 110, 20),
    (200, 60, 120),
    (110, 110, 110),
]
LINE_COLOR = (0, 40, 200)
BOX_COLORS = {"depth": (255, 0, 0), "lithology": (0, 160, 255), "legend": (255, 130, 0)}


# --------------------------------------------------------------------------------------------- CSV


def write_csv(intervals: Iterable[Interval], path: Path, *, with_diagnostics: bool = False) -> None:
    """Write the submission CSV (organisers' column order). Diagnostics add confidence and notes."""
    cols = list(CSV_COLUMNS) + (["confidence", "notes"] if with_diagnostics else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for iv in intervals:
            row = iv.csv_row()
            if with_diagnostics:
                row["confidence"] = round(iv.confidence, 3)
                row["notes"] = "; ".join(iv.notes)
            w.writerow(row)


def read_csv(path: Path) -> list[Interval]:
    """Read a results or label CSV. Accepts ``bottom_depth`` or ``base_depth``; cluster_id is optional."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            bottom = r.get("bottom_depth", r.get("base_depth"))
            cid = (r.get("cluster_id") or "").strip()
            out.append(
                Interval(
                    top_y=int(float(r["top_y"])),
                    bottom_y=int(float(r["bottom_y"])),
                    lithology=r["lithology"].strip(),
                    cluster_id=int(cid) if cid.lstrip("-").isdigit() else (cid or None),  # type: ignore[arg-type]
                    top_depth=float(r["top_depth"]) if r.get("top_depth") not in (None, "") else float("nan"),
                    bottom_depth=float(bottom) if bottom not in (None, "") else float("nan"),
                )
            )
    return out


# --------------------------------------------------------------------------------------------- image


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "arialbd.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_color_map(labels: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """A distinct colour per label, in order of first appearance; nothing is hard-coded per lithology."""
    colors: dict[str, tuple[int, int, int]] = {}
    for label in labels:
        if label in colors:
            continue
        i = len(colors)
        if i < len(PALETTE):
            colors[label] = PALETTE[i]
        else:  # beyond the palette: spread hues with the golden ratio so neighbours stay distinct
            h = (i * 0.618034) % 1.0
            colors[label] = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.8))  # type: ignore[assignment]
    return colors


def render_annotated(
    img: Image.Image,
    boxes: DocumentBoxes,
    intervals: list[Interval],
    *,
    max_height_px: int = 20000,
    coverage: float = 0.5,
    title: str = "",
) -> Image.Image:
    """Lithology column with predicted intervals, boundary depths and labels, plus a colour key on top.

    The view spans the depth column to the lithology column (with margins for labels) and is shrunk so it is at
    most ``max_height_px`` tall.
    """
    dc, lc = boxes.depth_column, boxes.lithology_column
    y0 = min(dc.y0, lc.y0)
    y1 = max(dc.y1, lc.y1)
    if intervals:
        y0, y1 = min(y0, intervals[0].top_y), max(y1, intervals[-1].bottom_y)
    margin_l, margin_r = 160, 260
    x0 = max(0, min(dc.x0, lc.x0) - margin_l)
    x1 = min(img.width, max(dc.x1, lc.x1) + margin_r)
    region = Box(x0, max(0, y0 - 40), x1, min(img.height, y1 + 40))
    scale = min(1.0, max_height_px / region.height)
    view = Image.fromarray(crop_rgb(img, region, scale))

    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(13)
    sx = lambda x: (x - region.x0) * scale  # noqa: E731
    sy = lambda y: (y - region.y0) * scale  # noqa: E731
    for b, name in ((dc, "depth"), (lc, "lithology")):
        draw.rectangle([sx(b.x0), sy(b.y0), sx(b.x1) - 1, sy(b.y1) - 1], outline=BOX_COLORS[name] + (255,), width=2)

    colors = build_color_map(iv.lithology for iv in intervals)
    lx0, lx1 = sx(lc.x0), sx(lc.x1)
    lxm = lx0 + (lx1 - lx0) * coverage
    xd = sx(dc.x0)
    last_label_y = -1e9
    for iv in intervals:
        ya, yb = sy(iv.top_y), sy(iv.bottom_y)
        c = colors[iv.lithology]
        draw.rectangle([lx0, ya, lxm, max(ya, yb - 1)], fill=c + (150,), outline=c + (255,), width=1)
        draw.line([(xd, ya), (lx1, ya)], fill=LINE_COLOR + (255,), width=1)
        if ya - last_label_y >= 15:  # skip labels that would overlap the previous one
            draw.text(
                (xd - 6, ya),
                f"{iv.top_depth:g}",
                fill=LINE_COLOR + (255,),
                font=font,
                anchor="rm",
                stroke_width=3,
                stroke_fill=(255, 255, 255, 255),
            )
            unsure = next((n for n in iv.notes if n.startswith("computer vision could not differentiate")), None)
            tag = ("CV couldn't differentiate: " + unsure.split(": ", 1)[1]) if unsure else iv.lithology
            tag += f" (c{iv.cluster_id})" if iv.cluster_id is not None else ""
            draw.text(
                (lx1 + 8, (ya + yb) / 2),
                tag,
                fill=c + (255,),
                font=font,
                anchor="lm",
                stroke_width=3,
                stroke_fill=(255, 255, 255, 255),
            )
            last_label_y = ya
    if intervals:
        yb = sy(intervals[-1].bottom_y)
        draw.line([(xd, yb), (lx1, yb)], fill=LINE_COLOR + (255,), width=1)
        draw.text(
            (xd - 6, yb),
            f"{intervals[-1].bottom_depth:g}",
            fill=LINE_COLOR + (255,),
            font=font,
            anchor="rm",
            stroke_width=3,
            stroke_fill=(255, 255, 255, 255),
        )
    view = Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")
    return _add_key(view, colors, title)


def _add_key(img: Image.Image, colors: dict[str, tuple[int, int, int]], title: str, row_h: int = 30) -> Image.Image:
    font = get_font(14)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    slots, x, row = [], 10, 1 if title else 0
    for name in colors:
        w = 26 + int(measure.textlength(name, font=font)) + 20
        if x + w > img.width and x > 10:
            x, row = 10, row + 1
        slots.append((name, x, row))
        x += w
    bar = row_h * (row + 1)
    out = Image.new("RGB", (img.width, img.height + bar), (255, 255, 255))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    if title:
        d.text((10, row_h / 2), title[:120], fill=(0, 0, 0), font=font, anchor="lm")
    for name, x, r in slots:
        y = r * row_h
        d.rectangle([x, y + 6, x + 18, y + 24], fill=colors[name], outline=(0, 0, 0))
        d.text((x + 26, y + 15), name, fill=(0, 0, 0), font=font, anchor="lm")
    return out
