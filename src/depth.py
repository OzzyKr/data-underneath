import time
import numpy as np
import pytesseract
from sklearn.linear_model import RANSACRegressor, LinearRegression
from src.data import load_crops

OCR_CFG = "--psm 11 -c tessedit_char_whitelist=0123456789"

def read_labels(depth_crop, y_offset, chunk=4000, overlap=400):
    """OCR depth labels. Returns [(page_y, value)]."""
    g = depth_crop.convert("L")
    W, H = g.size
    pts = []
    for y0 in range(0, H, chunk - overlap):
        part = g.crop((0, y0, W, min(H, y0 + chunk))).rotate(90, expand=True)  # labels are vertical
        d = pytesseract.image_to_data(part, config=OCR_CFG, output_type=pytesseract.Output.DICT)
        for txt, left, width, conf in zip(d["text"], d["left"], d["width"], d["conf"]):
            if txt.isdigit() and len(txt) >= 3 and float(conf) > 50:
                pts.append((y_offset + y0 + left + width / 2, int(txt)))
    return pts

def fit_depth(pts, tol_ft=20):
    """Robust linear fit page_y -> depth. Returns (a, b, n_inliers)."""
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