"""Group intervals that share a fill (same colour and pattern) without fixing the number of groups."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from lithology.config import ClusterConfig
from lithology.features import COLOUR_IDX, FeatureScaler


def weighted(desc_scaled: np.ndarray, colour_weight: float) -> np.ndarray:
    w = np.ones(desc_scaled.shape[1])
    w[COLOUR_IDX] = colour_weight
    return desc_scaled * w / np.sqrt((w**2).sum())  # RMS-style distance, comparable across feature counts


def cluster_intervals(desc: np.ndarray, scaler: FeatureScaler, cfg: ClusterConfig) -> np.ndarray:
    """Cluster ids (0..k-1, ordered by first appearance top-down) for interval descriptors (N, F)."""
    n = len(desc)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.zeros(1, int)
    X = weighted(scaler.transform(desc), cfg.colour_weight)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cfg.distance_threshold, linkage="average"
    ).fit_predict(X)
    # Renumber by first appearance so ids are stable and readable (deterministic, top-down).
    order: dict[int, int] = {}
    for lab in labels:
        order.setdefault(int(lab), len(order))
    return np.array([order[int(lab)] for lab in labels])


def merge_adjacent(tops: list[int], bottoms: list[int], ids: list[int]) -> list[tuple[int, int, int]]:
    """Merge touching intervals that ended up in the same cluster."""
    out: list[list[int]] = []
    for t, b, c in zip(tops, bottoms, ids, strict=True):
        if out and out[-1][2] == c and out[-1][1] == t:
            out[-1][1] = b
        else:
            out.append([t, b, c])
    return [(t, b, c) for t, b, c in out]
