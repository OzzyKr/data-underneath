"""Lithology column segmentation + clustering (numpy / OpenCV only)."""
import warnings
import cv2
import numpy as np


# ---------- small numpy replacements for sklearn / skimage ----------

def lbp_uniform(gray):
    """Uniform LBP (P=8, R=1) -> codes 0..9 (9 = non-uniform)."""
    g = gray.astype(np.int16)
    H, W = g.shape
    c = g[1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = [g[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx] >= c for dy, dx in offs]
    ones = np.zeros_like(c, dtype=np.uint8)
    trans = np.zeros_like(c, dtype=np.uint8)
    for i in range(8):
        ones += bits[i]
        trans += bits[i] != bits[(i + 1) % 8]
    out = np.full((H, W), 9, np.uint8)
    out[1:-1, 1:-1] = np.where(trans <= 2, ones, 9)
    return out


def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def kmeans(X, k, seed=0, attempts=4):
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, lab, cent = cv2.kmeans(X.astype(np.float32), k, None, crit, attempts, cv2.KMEANS_PP_CENTERS)
    return lab.ravel(), cent


def silhouette(X, lab):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    ks = np.unique(lab)
    if len(ks) < 2:
        return -1.0
    M = np.stack([D[:, lab == k].mean(1) for k in ks], 1)          # mean dist to each cluster
    sizes = np.array([(lab == k).sum() for k in ks])
    own = np.searchsorted(ks, lab)
    a = M[np.arange(len(X)), own] * sizes[own] / np.maximum(sizes[own] - 1, 1)
    M[np.arange(len(X)), own] = np.inf
    b = M.min(1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[sizes[own] == 1] = 0
    return float(s.mean())


# ---------- segmentation ----------

def ink_mask(arr):
    """Black pattern lines: dark AND unsaturated (so dark colour fills are not counted as ink)."""
    a = arr.astype(np.float32)
    gray = a.mean(axis=2)
    sat = (a.max(axis=2) - a.min(axis=2)) / 255
    return (gray < 110) & (sat < 0.25)


def _interior(score, W, thr=0.85, margin=3):
    """Widest gap between thick vertical border runs in one row-chunk, or None."""
    on = score > thr
    edges, x = [], 0
    while x < W:                                   # collect border runs
        if on[x]:
            a = x
            while x < W and on[x]:
                x += 1
            edges.append((a, x))
        x += 1
    if not edges:
        return None
    cuts = [0] + [e for r in edges for e in r] + [W]
    gaps = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    if b - a < 0.3 * W:
        return None
    return a + margin, b - margin


def normalize_column(arr, chunk=500):
    """Keep only the inside of the column's thick borders, chunk by chunk.
    Scanned logs drift sideways, so borders are found per chunk and every chunk
    is resized to the same width (rows are untouched, so y stays valid)."""
    ink = ink_mask(arr)
    H, W = ink.shape
    spans, prev = [], (0, W)
    for y in range(0, H, chunk):
        sp = _interior(ink[y:y + chunk].mean(0), W) or prev
        spans.append(sp)
        prev = sp
    tw = int(np.median([b - a for a, b in spans]))
    parts = [cv2.resize(np.ascontiguousarray(arr[y:y + chunk, a:b]), (tw, min(chunk, H - y)),
                        interpolation=cv2.INTER_AREA)
             for (a, b), y in zip(spans, range(0, H, chunk))]
    return np.vstack(parts)


def row_background(rgb, ink):
    """Mean colour of each row, ignoring the black pattern lines."""
    bg = np.where(ink[..., None], np.nan, rgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(bg, axis=1)          # fully-black rows -> nan
    return np.nan_to_num(out, nan=0.0)


def features(arr, win=80, step=8, core=16):
    """Feature vectors for windows sliding down a (H, W, 3) uint8 patch.
    Layout: LBP hist 0-9 | mean RGB 10-12 | saturation 13 | ink stats 14-16 | core background RGB 17-19."""
    gray = arr.mean(axis=2).astype(np.float32)
    ink_all = ink_mask(arr)
    lbp = lbp_uniform(gray.astype(np.uint8))
    rgb = arr.astype(np.float32) / 255
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    bg_row = row_background(rgb, ink_all)
    H = arr.shape[0]
    win = min(win, H)
    core = min(core, win)
    ys, feats, blank = [], [], []
    for y in range(0, max(1, H - win + 1), step):
        s = slice(y, y + win)
        c = slice(y + win // 2 - core // 2, y + win // 2 + core // 2)
        ink = ink_all[s]
        f = np.concatenate([
            np.bincount(lbp[s].ravel(), minlength=10) / lbp[s].size,
            rgb[s].reshape(-1, 3).mean(0), [sat[s].mean()],
            [ink.mean(), ink.mean(axis=1).std(), ink.mean(axis=0).std()],
            bg_row[c].mean(0),
        ])
        ys.append(y)
        feats.append(f)
        blank.append(ink.mean() < 0.01 and sat[s].mean() < 0.05)
    return np.array(ys), np.array(feats), np.array(blank)


# feature weights applied after standardisation
WEIGHTS = np.ones(20)
WEIGHTS[17:20] = 3.0      # background colour of the core is the strongest cue


def choose_k(X, kmax=10, sample=2000, seed=0, tol=0.9):
    """Largest k whose silhouette is within 90% of the best (silhouette alone favours too few clusters)."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    scores = {}
    for k in range(2, min(kmax, len(Xs) - 1) + 1):
        lab, _ = kmeans(Xs, k, seed)
        scores[k] = silhouette(Xs, lab)
    best = max(scores.values())
    return max(k for k, s in scores.items() if s >= tol * best)


def merge_short_runs(labels, min_len):
    """Absorb runs shorter than min_len windows into the longer neighbour."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        runs, start = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                runs.append((start, i))
                start = i
        for j, (a, b) in enumerate(runs):
            if b - a < min_len and len(runs) > 1:
                prev = runs[j - 1] if j > 0 else None
                nxt = runs[j + 1] if j + 1 < len(runs) else None
                tgt = max([r for r in (prev, nxt) if r], key=lambda r: r[1] - r[0])
                labels[a:b] = labels[tgt[0]]
                changed = True
                break
    return labels


def segment_column(arr, win=40, step=8, min_px=24, k=None):
    """Returns (intervals, centroids, patch) - patch is the border-free column the intervals refer to.
    intervals: [(top_px, bottom_px, cluster_id)] in crop coordinates, cluster_id -1 = no data.
    centroids: {cluster_id: mean raw feature vector} (used to match the legend)."""
    arr = normalize_column(arr)
    ys, X, blank = features(arr, win, step)
    labels = np.full(len(ys), -1)
    if (~blank).sum() > 10:
        Xn = standardize(X[~blank]) * WEIGHTS
        k = k or choose_k(Xn)
        labels[~blank], _ = kmeans(Xn, min(k, int((~blank).sum()) - 1))
    labels = merge_short_runs(labels, max(1, min_px // step))
    cents = {int(c): X[labels == c].mean(0) for c in np.unique(labels) if c >= 0}
    cents["_mu"], cents["_sd"] = X[~blank].mean(0), np.where(X[~blank].std(0) < 1e-9, 1.0, X[~blank].std(0))
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            top = ys[start] + win // 2 if start else 0
            bot = ys[i - 1] + win // 2 + step if i < len(labels) else arr.shape[0]
            out.append((int(top), int(bot), int(labels[start])))
            start = i
    return out, cents, arr
