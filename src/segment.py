import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def trim_borders(arr, margin=3):
    """Drop thick vertical border lines and anything outside them."""
    gray = arr.mean(axis=2)
    ink = (gray < 128).mean(axis=0)          # dark-pixel fraction per x
    W = arr.shape[1]
    b = np.where(ink > 0.9)[0]
    left, right = b[b < W / 2], b[b > W / 2]
    x0 = left.max() + 1 + margin if len(left) else 0
    x1 = right.min() - margin if len(right) else W
    return (arr[:, x0:x1], x0) if x1 - x0 > 0.3 * W else (arr, 0)


def row_background(rgb, ink):
    """Mean colour of each row, ignoring the black pattern lines."""
    import warnings
    bg = np.where(ink[..., None], np.nan, rgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(bg, axis=1)          # (H, 3); fully-black rows -> nan
    return np.nan_to_num(out, nan=0.0)


def window_features(arr, win=80, step=8, core=16):
    gray = arr.mean(axis=2).astype(np.float32)
    ink_all = gray < 128
    lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method="uniform").astype(np.int8)
    rgb = arr.astype(np.float32) / 255
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    bg_row = row_background(rgb, ink_all)
    H = arr.shape[0]
    ys, feats, blank = [], [], []
    for y in range(0, max(1, H - win + 1), step):
        s = slice(y, y + win)                                   # big window: texture
        c = slice(y + win // 2 - core // 2, y + win // 2 + core // 2)  # small core: colour
        ink = ink_all[s]
        f = np.concatenate([
            np.bincount(lbp[s].ravel(), minlength=10) / lbp[s].size,       # texture (0-9)
            rgb[s].reshape(-1, 3).mean(0), [sat[s].mean()],               # window colour (10-13)
            [ink.mean(), ink.mean(axis=1).std(), ink.mean(axis=0).std()],  # density / structure (14-16)
            bg_row[c].mean(0),                                             # background colour, core (17-19)
        ])
        ys.append(y)
        feats.append(f)
        blank.append(ink.mean() < 0.01 and sat[s].mean() < 0.05)
    return np.array(ys), np.array(feats), np.array(blank)


def choose_k(X, kmax=10, sample=3000, seed=0, tol=0.9):
    """Largest k whose silhouette is within 90% of the best (silhouette alone favours too few clusters)."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    scores = {}
    for k in range(2, min(kmax, len(Xs) - 1) + 1):
        lab = KMeans(k, n_init=4, random_state=seed).fit_predict(Xs)
        scores[k] = silhouette_score(Xs, lab)
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


def segment_column(arr, win=80, step=8, min_px=32):
    """Returns [(top_px, bottom_px, cluster_id)] in crop coordinates. cluster_id -1 = no data."""
    arr, _ = trim_borders(arr)
    ys, X, blank = window_features(arr, win, step)
    labels = np.full(len(ys), -1)
    if (~blank).sum() > 10:
        Xn = StandardScaler().fit_transform(X[~blank])
        Xn[:, 17:20] *= 3.0   # background colour of the core is the strongest cue
        k = choose_k(Xn)
        labels[~blank] = KMeans(k, n_init=4, random_state=0).fit_predict(Xn)
    labels = merge_short_runs(labels, max(1, min_px // step))
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            top = ys[start] + win // 2 if start else 0
            bot = ys[i - 1] + win // 2 + step if i < len(labels) else arr.shape[0]
            out.append((int(top), int(bot), int(labels[start])))
            start = i
    return out