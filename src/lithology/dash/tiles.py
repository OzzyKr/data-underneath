"""Cut each scan into small web tiles for the Image view.

Two tile sets per scan, both sliced into ``tile_h``-px WebP tiles so a 313,800 px scan scrolls smoothly:

* ``c*.webp`` - the lithology column alone, at full resolution: the main track of the Image view;
* ``t*.webp`` - the wider context (depth column to lithology column, same margins as ``render.render_annotated``),
  shrunk to at most ``max_width`` px wide: the optional "full scan" track;
* ``p*.webp`` - the whole page, uncropped, shrunk to at most 1100 px wide: the "whole page" view.

The legend and each legend swatch are saved alongside for the legend panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from lithology.io import open_image
from lithology.models import Box, DocumentBoxes

MARGIN_L, MARGIN_R = 160, 260  # as render.render_annotated
Image.MAX_IMAGE_PIXELS = None  # scans are legitimately huge


def view_box(boxes: DocumentBoxes, width: int, height: int) -> Box:
    dc, lc = boxes.depth_column, boxes.lithology_column
    x0 = max(0, min(dc.x0, lc.x0) - MARGIN_L)
    x1 = min(width, max(dc.x1, lc.x1) + MARGIN_R)
    y0 = max(0, min(dc.y0, lc.y0))
    y1 = min(height, max(dc.y1, lc.y1))
    return Box(x0, y0, x1, y1)


def _mode(img: Image.Image) -> str:
    return "L" if img.mode in ("1", "L", "I;16", "I") else "RGB"


def _column_tiles(
    img: Image.Image, boxes: DocumentBoxes, out_dir: Path, mode: str, *, tile_h: int, quality: int, max_width: int = 360
) -> dict[str, Any]:
    """The lithology column with a thin margin, at full resolution (narrow, so cheap)."""
    W, H = img.size
    lc = boxes.lithology_column.clip(W, H)
    pad = max(2, round(lc.width * 0.04))
    box = Box(max(0, lc.x0 - pad), lc.y0, min(W, lc.x1 + pad), lc.y1)
    scale = min(1.0, max_width / box.width)
    out_w = max(1, round(box.width * scale))
    out_h = max(1, round(box.height * scale))
    n = (out_h + tile_h - 1) // tile_h
    for k in range(n):
        ty0, ty1 = k * tile_h, min(out_h, (k + 1) * tile_h)
        sy0 = box.y0 + int(ty0 / scale)
        sy1 = min(box.y1, box.y0 + int(round(ty1 / scale)))
        part = img.crop((box.x0, sy0, box.x1, max(sy1, sy0 + 1))).convert(mode)
        if scale < 1 or part.height != ty1 - ty0:
            part = part.resize((out_w, ty1 - ty0), Image.Resampling.LANCZOS)
        part.save(out_dir / f"c{k:04d}.webp", "WEBP", quality=quality, method=4)
    return {"box": list(box.as_tuple()), "scale": scale, "w": out_w, "h": out_h, "tile_h": tile_h, "n": n}


def _page_tiles(
    img: Image.Image, out_dir: Path, mode: str, *, tile_h: int, max_width: int = 1100, quality: int = 72
) -> dict[str, Any]:
    """The whole scan, uncropped, at overview resolution."""
    W, H = img.size
    scale = min(1.0, max_width / W)
    out_w, out_h = max(1, round(W * scale)), max(1, round(H * scale))
    n = (out_h + tile_h - 1) // tile_h
    for k in range(n):
        ty0, ty1 = k * tile_h, min(out_h, (k + 1) * tile_h)
        sy0 = int(ty0 / scale)
        sy1 = min(H, int(round(ty1 / scale)))
        part = img.crop((0, sy0, W, max(sy1, sy0 + 1))).convert(mode)
        part = part.resize((out_w, ty1 - ty0), Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST)
        part.save(out_dir / f"p{k:04d}.webp", "WEBP", quality=quality, method=4)
    return {"scale": scale, "w": out_w, "h": out_h, "tile_h": tile_h, "n": n, "image": [W, H]}


def build_tiles(
    image_path: Path,
    boxes: DocumentBoxes,
    legend: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_width: int = 900,
    tile_h: int = 1024,
    quality: int = 80,
) -> dict[str, Any]:
    """Write tiles + legend crops into ``out_dir`` and return the tile metadata (also saved as meta.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = open_image(image_path)
    W, H = img.size
    vb = view_box(boxes, W, H)
    scale = min(1.0, max_width / vb.width)
    out_w = max(1, round(vb.width * scale))
    out_h = max(1, round(vb.height * scale))
    mode = _mode(img)
    n = (out_h + tile_h - 1) // tile_h
    for k in range(n):
        ty0, ty1 = k * tile_h, min(out_h, (k + 1) * tile_h)
        sy0 = vb.y0 + int(ty0 / scale)
        sy1 = min(vb.y1, vb.y0 + int(round(ty1 / scale)))
        part = img.crop((vb.x0, sy0, vb.x1, max(sy1, sy0 + 1))).convert(mode)
        part = part.resize((out_w, ty1 - ty0), Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST)
        part.save(out_dir / f"t{k:04d}.webp", "WEBP", quality=quality, method=4)

    column = _column_tiles(img, boxes, out_dir, mode, tile_h=tile_h, quality=quality)
    page = _page_tiles(img, out_dir, mode, tile_h=tile_h)

    legend_meta = None
    lg = boxes.legend.clip(W, H)
    if lg.width > 2 and lg.height > 2:
        ls = min(1.0, 1400 / lg.width)
        crop = img.crop(lg.as_tuple()).convert(mode)
        if ls < 1:
            crop = crop.resize((round(lg.width * ls), round(lg.height * ls)), Image.Resampling.LANCZOS)
        crop.save(out_dir / "legend.webp", "WEBP", quality=88, method=4)
        legend_meta = {"box": list(lg.as_tuple()), "w": crop.width, "h": crop.height}
    for i, e in enumerate(legend):
        x0, y0, x1, y1 = e["swatch"]
        sw = Box(x0, y0, x1, y1).clip(W, H)
        if sw.width < 2 or sw.height < 2:
            continue
        crop = img.crop(sw.as_tuple()).convert("RGB")
        if crop.width > 96:
            crop = crop.resize((96, max(1, round(crop.height * 96 / crop.width))), Image.Resampling.LANCZOS)
        crop.save(out_dir / f"sw{i:03d}.png")

    meta = {
        "view": list(vb.as_tuple()),
        "scale": scale,
        "w": out_w,
        "h": out_h,
        "tile_h": tile_h,
        "n": n,
        "legend": legend_meta,
        "col": column,
        "page": page,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta
