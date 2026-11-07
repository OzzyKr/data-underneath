import time
import numpy as np
import pytesseract
from sklearn.linear_model import RANSACRegressor, LinearRegression
from src.data import load_crops
from PIL import Image

OCR_CFG = "--psm 11 -c tessedit_char_whitelist=0123456789,"

def _ocr_hits(img):
    d = pytesseract.image_to_data(img, config=OCR_CFG, output_type=pytesseract.Output.DICT)
    out = []
    for t, l, tp, w, h, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
        t = t.replace(",", "")
        if t.isdigit() and len(t) >= 2 and float(c) > 50:
            out.append((int(t), l, tp, w, h))
    return out

def _rotate(part, rot):
    return part if rot == 0 else part.rotate(rot, expand=True)


def _page_y(rot, y0, part_h, l, tp, w, h):
    if rot == 0:
        return y0 + tp + h / 2           # horizontal labels
    if rot == 90:
        return y0 + l + w / 2            # text reads top -> bottom
    return y0 + part_h - (l + w / 2)     # rot == -90: bottom -> top


def read_labels(depth_crop, y_offset, chunk=4000, overlap=400):
    """OCR depth labels in any orientation. Returns [(page_y, value)]."""
    g = depth_crop.convert("L")
    W, H = g.size
    s = max(1.0, 150 / W)  # upscale narrow columns so digits are big enough for OCR

    def prep(y0):
        part = g.crop((0, y0, W, min(H, y0 + chunk)))
        if s > 1:
            part = part.resize((int(part.width * s), int(part.height * s)), Image.LANCZOS)
        return part

    starts = list(range(0, H, chunk - overlap))
    sample = starts[::max(1, len(starts) // 3)][:3]
    best = max((0, 90, -90), key=lambda r: sum(
        len(_ocr_hits(_rotate(prep(y0), r))) for y0 in sample))
    pts = []
    for y0 in starts:
        part = prep(y0)
        for v, l, tp, w, h in _ocr_hits(_rotate(part, best)):
            y_in_part = _page_y(best, 0, part.height, l, tp, w, h) / s
            pts.append((y_offset + y0 + y_in_part, v))
    return pts


def fit_depth(pts, tol_ft=20):
    """Robust linear fit page_y -> depth. Returns (a, b, n_inliers)."""
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} depth labels read by OCR")
    y = np.array([p[0] for p in pts]).reshape(-1, 1)
    v = np.array([p[1] for p in pts], dtype=float)
    m = RANSACRegressor(LinearRegression(), residual_threshold=tol_ft).fit(y, v)
    return m.estimator_.coef_[0], m.estimator_.intercept_, int(m.inlier_mask_.sum())


def depth_mapping(ann, crops):
    y1 = ann["bounding_boxes"]["depth_column"][1]
    pts = read_labels(crops["depth_column"], y1)
    a, b, n = fit_depth(pts)
    return (lambda y: a * y + b), dict(a=a, b=b, n_labels=len(pts), n_inliers=n)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    name = sys.argv[1] if len(sys.argv) > 1 else \
        sorted(Path("dataset/bbox_annotation").glob("*.json"))[0].stem
    t = time.time()
    ann, crops = load_crops(name)
    to_depth, info = depth_mapping(ann, crops)
    box = ann["bounding_boxes"]["interpreted_lithology_column"]
    print(info)
    print(f"litho column: {to_depth(box[1]):.0f} ft -> {to_depth(box[3]):.0f} ft")
    print(f"{time.time() - t:.1f}s")