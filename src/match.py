"""Match column clusters to legend swatches with scale-tolerant pattern descriptors + fill colour."""
import cv2
import numpy as np
from src.segment import ink_mask, row_background

TEX_W = np.array([0.5, 2.0, 2.0, 2.0, 0.2, 1.0])   # anisotropy, dash/dot/vertical blob fractions, log ink, long lines
COL_W = 6.0                               # background colour weight


def describe(patch):
    """Texture (scale-tolerant) + background colour of an RGB uint8 patch."""
    ink = ink_mask(patch)
    H, W = ink.shape
    rs, cs, fr = ink.mean(1).std(), ink.mean(0).std(), ink.mean()
    n, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    st = st[1:]
    st = st[st[:, 4] >= 3]
    if len(st):
        st = st[st[:, 4] >= 0.2 * np.median(st[:, 4])]          # ignore specks
        la = np.log((st[:, 2] + 1) / (st[:, 3] + 1))
        dash, vert = float((la > 0.7).mean()), float((la < -0.7).mean())
        dot = 1.0 - dash - vert
        longh = float((st[:, 2] > 0.6 * W).mean())
    else:
        dash = dot = vert = longh = 0.0
    tex = np.array([np.log((rs + 0.01) / (cs + 0.01)), dash, dot, vert, np.log(fr + 0.005), longh])
    rgb = patch.astype(np.float32) / 255
    bg = row_background(rgb, ink).mean(0)
    return tex, bg


def distance(a, b):
    (ta, ca), (tb, cb) = a, b
    return float(np.linalg.norm((ta - tb) * TEX_W) + COL_W * np.linalg.norm(ca - cb))


def cluster_descriptors(intervals, patch, win=40, max_samples=30):
    """Median descriptor of each cluster, from the middle of its largest intervals."""
    by = {}
    for t, b, c in intervals:
        if c >= 0:
            by.setdefault(c, []).append((b - t, t, b))
    out = {}
    for c, ivs in by.items():
        descs = []
        for h, t, b in sorted(ivs, reverse=True)[:max_samples]:
            m = (t + b) // 2
            hh = max(8, min(win, h) // 2)
            p = patch[max(0, m - hh):m + hh]
            if p.shape[0] >= 8:
                descs.append(describe(p))
        if descs:
            out[c] = (np.median([d[0] for d in descs], 0), np.median([d[1] for d in descs], 0))
    return out


def legend_descriptors(legend_img, swatches):
    a = np.array(legend_img.convert("RGB"))
    out = []
    for s in swatches:
        x0, y0, x1, y1 = s["box"]
        px, py = max(1, (x1 - x0) // 7), max(1, (y1 - y0) // 7)
        out.append(describe(a[y0 + py:y1 - py, x0 + px:x1 - px]))
    return out


def merged_names(swatches, sdesc, eps=0.15):
    """Swatches that look identical (e.g. Clay vs Shale) share one name 'Clay/Shale'."""
    names = [s["name"] for s in swatches]
    out = list(names)
    for i in range(len(names)):
        same = sorted({names[j] for j in range(len(names)) if distance(sdesc[i], sdesc[j]) < eps})
        out[i] = "/".join(same)
    return out


def assign(intervals, patch, legend_img, swatches):
    """Returns {cluster_id: (swatch_index, name, distance)}."""
    if not swatches:
        return {}
    sdesc = legend_descriptors(legend_img, swatches)
    names = merged_names(swatches, sdesc)
    res = {}
    for c, d in cluster_descriptors(intervals, patch).items():
        dist = [distance(d, s) for s in sdesc]
        i = int(np.argmin(dist))
        res[c] = (i, names[i], dist[i])
    return res
