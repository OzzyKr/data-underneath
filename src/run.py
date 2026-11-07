import csv, sys, time
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from src.data import load_crops, ROOT
from src.depth import depth_mapping
from src.segment import segment_column

FIELDS = ["top_y", "bottom_y", "lithology", "cluster_id", "top_depth", "base_depth"]
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (128, 128, 0)]


def process(name, out_dir):
    t = time.time()
    ann, crops = load_crops(name)
    to_depth, _ = depth_mapping(ann, crops)
    y1 = ann["bounding_boxes"]["interpreted_lithology_column"][1]
    col = crops["interpreted_lithology_column"]
    rows = []
    for top, bot, cid in segment_column(np.array(col)):
        ty, by = y1 + top, y1 + bot
        rows.append(dict(top_y=ty, bottom_y=by,
                         lithology="No data" if cid < 0 else f"cluster_{cid}",  # legend matching comes next
                         cluster_id=cid,
                         top_depth=round(float(to_depth(ty)), 1),
                         base_depth=round(float(to_depth(by)), 1)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    img = col.copy()
    d = ImageDraw.Draw(img)
    for r in rows:
        y0, yb = r["top_y"] - y1, r["bottom_y"] - y1
        c = (150, 150, 150) if r["cluster_id"] < 0 else PALETTE[r["cluster_id"] % len(PALETTE)]
        d.rectangle([img.width - 25, y0, img.width, yb], fill=c)
        d.line([(0, y0), (img.width, y0)], fill=c, width=2)
    img.save(out_dir / f"{name}_annotated.png")
    print(f"{name[:40]}: {len(rows)} intervals, {time.time() - t:.1f}s")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "bbox_annotation").glob("*.json"))
    for n in names:
        try:
            process(n, Path("outputs/results"))
        except Exception as e:
            print(f"{n[:40]}: FAILED ({e})")