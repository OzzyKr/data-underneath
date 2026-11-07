import json
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # logs are ~450M pixels
ROOT = Path("dataset")

def load_annotation(name):
    with open(ROOT / "bbox_annotation" / f"{Path(name).stem}.json") as f:
        return json.load(f)

def load_crops(name):
    ann = load_annotation(name)
    img = Image.open(ROOT / "images" / ann["image_name"])
    crops = {k: img.crop(tuple(b)).convert("RGB")
             for k, b in ann["bounding_boxes"].items() if b}
    return ann, crops

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else \
        sorted((ROOT / "bbox_annotation").glob("*.json"))[0].stem
    ann, crops = load_crops(name)
    out = Path("outputs/crops"); out.mkdir(parents=True, exist_ok=True)
    for k, c in crops.items():
        c.save(out / f"{name}_{k}.png")
        print(k, c.size)