"""Hand-labelling kit: pictures + pre-filled draft CSVs a person corrects, and the import back to label files.

The person edits only depths and rock names (what they can read off the log). Pixel rows are recomputed from the
document's depth scale on import, so labels stay in the organisers' format (top_y/bottom_y + depths + name).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lithology.config import Config
from lithology.io import Document, crop_rgb, open_image
from lithology.models import Box, Interval
from lithology.pipeline import process
from lithology.render import get_font, write_csv

KIT_COLUMNS = ("n", "top_depth", "bottom_depth", "lithology", "note")


@dataclass(frozen=True)
class KitItem:
    key: str  # short file prefix, e.g. "03_rio"
    doc_id: str
    split: str  # dev | holdout
    window_top_ft: float
    window_bottom_ft: float
    knots_y: list[float]
    knots_depth: list[float]
    column_y0: int
    column_y1: int
    draft_sha256: str = ""  # hash of the draft CSV as generated; a different file means the person edited it


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def is_edited(item: KitItem, out_dir: Path) -> bool:
    """True when the person changed the draft CSV since it was generated (or its hash was never recorded)."""
    current = file_sha256(out_dir / f"{item.key}.csv")
    return bool(current) and current != item.draft_sha256


def choose_window(intervals: list[Interval], window_ft: float) -> tuple[float, float]:
    """The window (top, bottom depth) with the most distinct rock groups, then the most boundaries."""
    ivs = [iv for iv in intervals if math.isfinite(iv.top_depth) and math.isfinite(iv.bottom_depth)]
    if not ivs:
        raise ValueError("no intervals with depths to choose a window from")
    lo, hi = ivs[0].top_depth, ivs[-1].bottom_depth
    if hi - lo <= window_ft:
        return lo, hi
    best, best_score = (lo, lo + window_ft), (-1, -1)
    step = max(10.0, window_ft / 10)
    top = lo
    while top + window_ft <= hi:
        inside = [iv for iv in ivs if iv.bottom_depth > top and iv.top_depth < top + window_ft]
        score = (len({iv.cluster_id for iv in inside}), len(inside))
        if score > best_score:
            best, best_score = (top, top + window_ft), score
        top += step
    return best


def _y_of(depth: float, knots_y: list[float], knots_depth: list[float]) -> float:
    """Inverse of the piecewise-linear depth scale (depth increases with y)."""
    ys, ds = np.array(knots_y), np.array(knots_depth)
    if depth < ds[0] or depth > ds[-1]:
        i = 0 if depth < ds[0] else len(ds) - 2
        return float(ys[i] + (depth - ds[i]) * (ys[i + 1] - ys[i]) / (ds[i + 1] - ds[i]))
    return float(np.interp(depth, ds, ys))


def make_item(
    doc: Document, key: str, split: str, cfg: Config, out_dir: Path, window: tuple[float, float] | None = None
) -> KitItem:
    res = process(doc, cfg)
    if not res.ok or res.scale is None or res.boxes is None:
        raise RuntimeError(f"{doc.doc_id}: cannot build a kit item ({res.error or 'no depth scale'})")
    sc = res.scale
    window_ft = 200.0 if sc.px_per_ft >= 5 else 400.0
    top_ft, bot_ft = window if window else choose_window(res.intervals, window_ft)
    knots_y, knots_d = list(sc.knots_y), list(sc.knots_depth)
    lc, dc = res.boxes.lithology_column, res.boxes.depth_column
    y0 = max(lc.y0, int(_y_of(top_ft, knots_y, knots_d)))
    y1 = min(lc.y1, int(math.ceil(_y_of(bot_ft, knots_y, knots_d))))

    img = open_image(doc.image_path)
    # Show depth column, lithology column and the description text just right of it.
    x0 = max(0, min(dc.x0, lc.x0) - 20)
    x1 = min(img.width, max(dc.x1, lc.x1) + int(3.5 * lc.width) + 120)
    crop = Image.fromarray(crop_rgb(img, Box(x0, y0, x1, y1)))
    zoom = 1.0 if sc.px_per_ft >= 5 else 5.0 / sc.px_per_ft  # small scales: enlarge so 1 ft >= ~5 px
    zoom = max(1.0, min(zoom, 1500 / crop.width))  # but keep the picture a sensible width
    if zoom != 1.0:
        crop = crop.resize((round(crop.width * zoom), round(crop.height * zoom)), Image.Resampling.NEAREST)
    margin = 330
    canvas = Image.new("RGB", (crop.width + margin, crop.height + 40), (255, 255, 255))
    canvas.paste(crop, (0, 20))
    d = ImageDraw.Draw(canvas)
    font, small = get_font(16), get_font(13)

    def ypx(depth: float) -> float:
        return (_y_of(depth, knots_y, knots_d) - y0) * zoom + 20

    lx0, lx1 = (lc.x0 - x0) * zoom, (lc.x1 - x0) * zoom
    # depth ruler every 10 ft (5 ft minor) in the right margin
    rx = crop.width + 10
    first = math.ceil(top_ft / 5) * 5
    dep = first
    while dep <= bot_ft:
        yy = ypx(dep)
        major = dep % 10 == 0
        d.line([(rx, yy), (rx + (24 if major else 12), yy)], fill=(0, 90, 200), width=2 if major else 1)
        if major:
            d.text((rx + 30, yy), f"{dep:g} ft", fill=(0, 90, 200), font=small, anchor="lm")
        dep += 5
    # draft boundaries: red ticks outside the column on both sides + interval number and draft name
    rows = []
    inside = [iv for iv in res.intervals if iv.bottom_depth > top_ft and iv.top_depth < bot_ft]
    for n, iv in enumerate(inside, start=1):
        t, b = max(iv.top_depth, top_ft), min(iv.bottom_depth, bot_ft)
        ya, yb = ypx(t), ypx(b)
        for yy in (ya, yb):
            d.line([(lx0 - 28, yy), (lx0 - 3, yy)], fill=(220, 0, 0), width=3)
            d.line([(lx1 + 3, yy), (lx1 + 28, yy)], fill=(220, 0, 0), width=3)
        d.text((rx + 120, (ya + yb) / 2), f"#{n} {iv.lithology}"[:28], fill=(200, 0, 0), font=font, anchor="lm")
        rows.append(
            {"n": n, "top_depth": round(t, 1), "bottom_depth": round(b, 1), "lithology": iv.lithology, "note": ""}
        )
    d.text((10, 2), f"{key}  |  {doc.doc_id[:80]}  |  {top_ft:.0f}-{bot_ft:.0f} ft", fill=(0, 0, 0), font=small)
    canvas.save(out_dir / f"{key}.png")

    lg = res.boxes.legend.clip(*img.size)
    leg = Image.fromarray(crop_rgb(img, lg))
    if leg.width > 1600:
        leg = leg.resize((1600, round(leg.height * 1600 / leg.width)), Image.Resampling.LANCZOS)
    leg.save(out_dir / f"{key}_legend.png")
    names = sorted({e.label for e in res.legend})
    (out_dir / f"{key}_legend_names.txt").write_text("\n".join(names) + "\n")

    with open(out_dir / f"{key}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KIT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return KitItem(
        key,
        doc.doc_id,
        split,
        round(top_ft, 1),
        round(bot_ft, 1),
        knots_y,
        knots_d,
        lc.y0,
        lc.y1,
        file_sha256(out_dir / f"{key}.csv"),
    )


def save_manifest(items: list[KitItem], out_dir: Path) -> None:
    (out_dir / "kit.json").write_text(json.dumps([i.__dict__ for i in items], indent=2))


def import_kit(kit_dir: Path, labels_dir: Path) -> list[str]:
    """Convert corrected kit CSVs into label files (organisers' format). Returns a list of problems found."""
    items = [KitItem(**d) for d in json.loads((kit_dir / "kit.json").read_text())]
    problems: list[str] = []
    split: dict[str, str] = {}
    for it in items:
        with open(kit_dir / f"{it.key}.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
        ivs: list[Interval] = []
        prev_bottom = None
        for r in rows:
            try:
                t, b = float(r["top_depth"]), float(r["bottom_depth"])
            except (TypeError, ValueError):
                problems.append(f"{it.key} row {r.get('n')}: depths must be numbers")
                continue
            name = (r.get("lithology") or "").strip()
            if not name:
                problems.append(f"{it.key} row {r.get('n')}: missing lithology name")
            if b <= t:
                problems.append(f"{it.key} row {r.get('n')}: bottom_depth must be below top_depth")
                continue
            if prev_bottom is not None and abs(t - prev_bottom) > 0.05:
                problems.append(f"{it.key} row {r.get('n')}: top {t} does not meet previous bottom {prev_bottom}")
            prev_bottom = b
            ya = round(_y_of(t, it.knots_y, it.knots_depth))
            yb = round(_y_of(b, it.knots_y, it.knots_depth))
            if yb > ya:
                ivs.append(Interval(top_y=ya, bottom_y=yb, lithology=name or "Unknown", top_depth=t, bottom_depth=b))
        if ivs:
            write_csv(ivs, labels_dir / f"{it.doc_id}.csv")
            split[it.doc_id] = it.split
    _update_split(labels_dir, split)
    return problems


def _update_split(labels_dir: Path, new: dict[str, str]) -> None:
    import yaml

    path = labels_dir / "manifest.yaml"
    header = "".join(line for line in path.read_text().splitlines(keepends=True) if line.startswith("#"))
    data = yaml.safe_load(path.read_text()) or {}
    split = dict(data.get("split") or {})
    split.update(new)
    path.write_text(header + yaml.safe_dump({"split": dict(sorted(split.items()))}, sort_keys=False))
