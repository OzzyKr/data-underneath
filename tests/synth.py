"""Synthetic well-log generator with exact ground truth, so every step is testable without the real dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# name -> (fill colour, pattern kind)
ROCKS: dict[str, tuple[tuple[int, int, int], str]] = {
    "Sandstone": ((255, 235, 60), "dots"),
    "Shale": ((60, 170, 70), "hdash"),
    "Limestone": ((130, 210, 245), "brick"),
    "Siltstone": ((200, 200, 200), "vbar"),
    "Coal": ((20, 20, 20), "solid"),
}


def paint(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, rock: str, phase: int = 0) -> None:
    """Fill arr[y0:y1, x0:x1] with a rock's colour and pattern (pattern anchored to absolute rows)."""
    colour, kind = ROCKS[rock]
    arr[y0:y1, x0:x1] = colour
    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    if kind == "dots":
        m = ((ys + phase) % 8 < 2) & (xs % 8 < 2)
    elif kind == "hdash":
        row = (ys + phase) % 10 < 2
        m = row & (((xs + ((ys + phase) // 10 % 2) * 7) % 14) < 8)
    elif kind == "vbar":
        m = ((ys + phase) % 14 < 8) & (xs % 10 < 2)
    elif kind == "brick":
        m = ((ys + phase) % 10 == 0) | (((xs + ((ys + phase) // 10 % 2) * 10) % 20) == 0)
    else:  # solid
        m = np.zeros((y1 - y0, x1 - x0), bool)
    region = arr[y0:y1, x0:x1]
    region[np.broadcast_to(m, region.shape[:2])] = (0, 0, 0)


@dataclass
class SynthDoc:
    image: Image.Image
    annotation: dict
    truth: list[tuple[int, int, str, float, float]]  # (top_y, bottom_y, rock, top_depth, bottom_depth)
    px_per_ft: float
    legend_names: list[str]


def make_document(
    intervals: list[tuple[float, str]],
    *,
    top_depth: float = 5000.0,
    px_per_ft: float = 10.0,
    label_every_ft: float = 20.0,
    rotate_labels: bool = False,
    legend_rocks: list[str] | None = None,
    legend_layout: str = "right",
) -> SynthDoc:
    """Build a log: legend in the header, depth column with printed labels, lithology column with intervals.

    ``intervals`` is a list of (thickness_ft, rock) from the top down.
    """
    legend_rocks = legend_rocks or list(ROCKS)
    total_ft = sum(t for t, _ in intervals)
    col_top = 420
    height = col_top + int(round(total_ft * px_per_ft)) + 60
    width = 900
    arr = np.full((height, width, 3), 255, np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)

    # legend: swatch + label, in a grid
    lg_box = [20, 20, 880, 380]
    arr = np.asarray(img).copy()
    swatch_w, swatch_h = 90, 46
    positions = []
    for i, rock in enumerate(legend_rocks):
        if legend_layout == "right":
            cx, cy = 50 + (i % 3) * 280, 60 + (i // 3) * 100
        else:
            cx, cy = 50 + (i % 4) * 210, 50 + (i // 4) * 150
        paint(arr, cy, cy + swatch_h, cx, cx + swatch_w, rock)
        arr[cy, cx : cx + swatch_w] = 0
        arr[cy + swatch_h - 1, cx : cx + swatch_w] = 0
        arr[cy : cy + swatch_h, cx] = 0
        arr[cy : cy + swatch_h, cx + swatch_w - 1] = 0
        positions.append((cx, cy, rock))

    # lithology column
    lith_x0, lith_x1 = 520, 580
    truth = []
    y = col_top
    d = top_depth
    for thick, rock in intervals:
        y_end = col_top + int(round((d + thick - top_depth) * px_per_ft))
        paint(arr, y, y_end, lith_x0, lith_x1, rock)
        truth.append((y, y_end, rock, d, d + thick))
        y, d = y_end, d + thick
    arr[col_top:y, lith_x0] = 0  # column borders
    arr[col_top:y, lith_x1 - 1] = 0
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    for cx, cy, rock in positions:
        if legend_layout == "right":
            draw.text((cx + swatch_w + 16, cy + swatch_h / 2), rock, fill=(0, 0, 0), font=font, anchor="lm")
        else:
            draw.text((cx + swatch_w / 2, cy + swatch_h + 18), rock, fill=(0, 0, 0), font=font, anchor="mm")

    # depth column with ticks and labels
    dc_x0, dc_x1 = 300, 400
    dep = top_depth
    while dep <= top_depth + total_ft:
        yy = col_top + (dep - top_depth) * px_per_ft
        draw.line([(dc_x1 - 12, yy), (dc_x1, yy)], fill=(0, 0, 0), width=2)
        text = f"{int(dep)}"
        if rotate_labels:
            tw = int(draw.textlength(text, font=font)) + 8
            tile = Image.new("RGB", (tw, 30), (255, 255, 255))
            ImageDraw.Draw(tile).text((tw / 2, 15), text, fill=(0, 0, 0), font=font, anchor="mm")
            tile = tile.rotate(-90, expand=True)
            img.paste(tile, (int((dc_x0 + dc_x1) / 2 - tile.width / 2), int(yy - tile.height / 2)))
        else:
            draw.text(((dc_x0 + dc_x1) / 2 - 6, yy), text, fill=(0, 0, 0), font=font, anchor="mm")
        dep += label_every_ft

    annotation = {
        "image_name": "synthetic.tif",
        "width": width,
        "height": height,
        "bbox_format": "xyxy",
        "bounding_boxes": {
            "depth_column": [dc_x0, col_top, dc_x1, y],
            "interpreted_lithology_column": [lith_x0, col_top, lith_x1, y],
            "lithology_legend": lg_box,
        },
    }
    return SynthDoc(img, annotation, truth, px_per_ft, legend_rocks)
