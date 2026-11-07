"""Full pipeline: scanned log -> intervals CSV + annotated column image.

python -m src.run                 # all logs in dataset/
python -m src.run NAME [NAME...]  # selected logs
"""
import csv
import string
import sys
import time
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from src.data import load_crops, ROOT
from src.depth import depth_mapping
from src.legend import parse_legend
from src.match import assign
from src.segment import segment_column

FIELDS = ["top_y", "bottom_y", "lithology", "cluster_id", "top_depth", "base_depth"]
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40)]


def label_intervals(intervals, mapping):
    """Replace raw cluster ids by legend names, merge neighbours with the same name,
    and give each distinct lithology a letter id (A, B, ...) in order of appearance."""
    named = []
    for t, b, c in intervals:
        name = "No data" if c < 0 else (mapping[c][1] if c in mapping else f"cluster_{c}")
        if named and named[-1][2] == name:
            named[-1][1] = b
        else:
            named.append([t, b, name])
    ids = {}
    for _, _, name in named:
        if name != "No data" and name not in ids:
            ids[name] = string.ascii_uppercase[len(ids) % 26]
    return [(t, b, name, ids.get(name, "-")) for t, b, name in named]


def annotate(col, rows, y1):
    img = col.copy()
    d = ImageDraw.Draw(img)
    letters = sorted({r["cluster_id"] for r in rows if r["cluster_id"] != "-"})
    colour = {L: PALETTE[i % len(PALETTE)] for i, L in enumerate(letters)}
    for r in rows:
        y0, yb = r["top_y"] - y1, r["bottom_y"] - y1
        c = colour.get(r["cluster_id"], (150, 150, 150))
        d.rectangle([img.width - 25, y0, img.width, yb], fill=c)
        d.line([(0, y0), (img.width, y0)], fill=c, width=2)
        if yb - y0 > 14:
            d.text((4, y0 + 2), f"{r['cluster_id']} {r['lithology'][:18]} {r['top_depth']:.0f}", fill=(255, 0, 0))
    return img


def process(name, out_dir):
    t0 = time.time()
    ann, crops = load_crops(name)
    to_depth, dinfo = depth_mapping(ann, crops)
    y1 = ann["bounding_boxes"]["interpreted_lithology_column"][1]
    col = crops["interpreted_lithology_column"]
    swatches = parse_legend(crops["lithology_legend"]) if "lithology_legend" in crops else []
    k = min(10, max(4, len(swatches))) if swatches else None
    intervals, _, patch = segment_column(np.array(col), k=k)
    mapping = assign(intervals, patch, crops["lithology_legend"], swatches) if swatches else {}
    rows = []
    for t, b, lith, cid in label_intervals(intervals, mapping):
        ty, by = y1 + t, y1 + b
        rows.append(dict(top_y=ty, bottom_y=by, lithology=lith, cluster_id=cid,
                         top_depth=round(float(to_depth(ty)), 1), base_depth=round(float(to_depth(by)), 1)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    annotate(col, rows, y1).save(out_dir / f"{name}_annotated.png")
    dt = time.time() - t0
    print(f"{name[:45]:45s} {len(rows):4d} intervals  {len(swatches):2d} swatches  "
          f"{dinfo['n_inliers']}/{dinfo['n_labels']} depth labels  {dt:5.1f}s", flush=True)
    return dt


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "bbox_annotation").glob("*.json"))
    out = Path("outputs/results")
    total, fails = 0.0, []
    for n in names:
        try:
            total += process(n, out)
        except Exception as e:
            fails.append(n)
            print(f"{n[:45]:45s} FAILED: {e}", flush=True)
    print(f"\n{len(names) - len(fails)}/{len(names)} logs processed, {total:.0f}s total, "
          f"{total / max(1, len(names) - len(fails)):.1f}s per log")
