import sys
from pathlib import Path
from PIL import ImageDraw
from src.data import load_crops
from src.depth import depth_mapping

name, start_ft, length_ft = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ann, crops = load_crops(name)
to_depth, info = depth_mapping(ann, crops)
y1 = ann["bounding_boxes"]["interpreted_lithology_column"][1]
a, b = info["a"], info["b"]
to_y = lambda d: (d - b) / a                      # depth -> page y
top, bot = int(to_y(start_ft) - y1), int(to_y(start_ft + length_ft) - y1)
col = crops["interpreted_lithology_column"].crop((0, top, crops["interpreted_lithology_column"].width, bot))
img = col.convert("RGB").resize((col.width * 2, col.height))   # wider = room for labels
d = ImageDraw.Draw(img)
ft = start_ft - start_ft % 10 + 10
while ft < start_ft + length_ft:
    y = to_y(ft) - y1 - top
    d.line([(0, y), (40, y)], fill=(255, 0, 0), width=2)
    d.text((44, y - 6), f"{ft:.0f}", fill=(255, 0, 0))
    ft += 10
out = Path("outputs/gt"); out.mkdir(parents=True, exist_ok=True)
img.save(out / f"{name}_{start_ft:.0f}.png")
(out / f"{name}.csv").write_text("top_depth,base_depth,lithology\n") if not (out / f"{name}.csv").exists() else None
print("saved", out / f"{name}_{start_ft:.0f}.png")