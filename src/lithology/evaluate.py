"""Metrics comparing predicted intervals with hand labels, plus leave-one-family-out grouping.

A label file may cover only a depth window of a log; every metric is computed inside the labelled extent
[first label top_y, last label bottom_y). Distances are converted to ft with the labels' own depths.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score

from lithology.models import Interval

UNLABELLED = "__none__"


def normalise_name(name: str) -> str:
    """Compare lithology names case- and punctuation-insensitively ("Sand-SS" == "sand ss")."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


@dataclass(frozen=True)
class DocScore:
    labelled_ft: float
    lithology_accuracy: float  # share of labelled feet whose predicted lithology matches
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    depth_mae_ft: float  # |predicted depth - labelled depth| at labelled boundaries
    cluster_ari: float  # adjusted Rand index between predicted clusters and labelled lithologies (per row)
    n_label_boundaries: int
    n_pred_boundaries: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _rows_to_labels(intervals: list[Interval], y0: int, y1: int, key) -> np.ndarray:
    out = np.full(y1 - y0, UNLABELLED, dtype=object)
    for iv in intervals:
        a, b = max(iv.top_y, y0), min(iv.bottom_y, y1)
        if b > a:
            out[a - y0 : b - y0] = key(iv)
    return out


def _depth_fn(intervals: list[Interval]):
    pts = sorted({(iv.top_y, iv.top_depth) for iv in intervals} | {(iv.bottom_y, iv.bottom_depth) for iv in intervals})
    pts = [(y, d) for y, d in pts if np.isfinite(d)]
    if len(pts) < 2:
        return lambda y: float("nan")
    ys = np.array([p[0] for p in pts], dtype=float)
    ds = np.array([p[1] for p in pts], dtype=float)

    def f(y: float) -> float:
        if y < ys[0] or y > ys[-1]:  # extrapolate linearly from the end segments
            i = 0 if y < ys[0] else len(ys) - 2
            return float(ds[i] + (ds[i + 1] - ds[i]) / (ys[i + 1] - ys[i]) * (y - ys[i]))
        return float(np.interp(y, ys, ds))

    return f


def _match(truth: list[int], pred: list[int], tol: float) -> int:
    """Greedy one-to-one matching of boundary rows within ``tol`` px, closest pairs first."""
    pairs = sorted((abs(t - p), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred) if abs(t - p) <= tol)
    used_t, used_p, hits = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            hits += 1
    return hits


def score_document(pred: list[Interval], label: list[Interval], tolerance_ft: float = 2.0) -> DocScore:
    if not label:
        raise ValueError("label has no intervals")
    label = sorted(label, key=lambda iv: iv.top_y)
    pred = sorted(pred, key=lambda iv: iv.top_y)
    y0, y1 = label[0].top_y, label[-1].bottom_y
    ft_per_px = (label[-1].bottom_depth - label[0].top_depth) / (y1 - y0)
    tol_px = tolerance_ft / ft_per_px

    truth_rows = _rows_to_labels(label, y0, y1, lambda iv: normalise_name(iv.lithology))
    pred_rows = _rows_to_labels(pred, y0, y1, lambda iv: normalise_name(iv.lithology))
    labelled = truth_rows != UNLABELLED
    accuracy = float(np.mean(pred_rows[labelled] == truth_rows[labelled])) if labelled.any() else float("nan")

    truth_b = [iv.top_y for iv in label[1:]]
    pred_b = [iv.top_y for iv in pred if y0 + tol_px < iv.top_y < y1 - tol_px]
    hits = _match(truth_b, pred_b, tol_px)
    precision = hits / len(pred_b) if pred_b else (1.0 if not truth_b else 0.0)
    recall = hits / len(truth_b) if truth_b else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    depth_of = _depth_fn(pred)
    errs = [abs(depth_of(iv.top_y) - iv.top_depth) for iv in label] + [abs(depth_of(y1) - label[-1].bottom_depth)]
    errs = [e for e in errs if np.isfinite(e)]
    mae = float(np.mean(errs)) if errs else float("nan")

    cluster_rows = _rows_to_labels(pred, y0, y1, lambda iv: f"c{iv.cluster_id}")
    ari = float(adjusted_rand_score(truth_rows[labelled], cluster_rows[labelled])) if labelled.any() else float("nan")

    return DocScore(
        labelled_ft=(y1 - y0) * ft_per_px,
        lithology_accuracy=accuracy,
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        depth_mae_ft=mae,
        cluster_ari=ari,
        n_label_boundaries=len(truth_b),
        n_pred_boundaries=len(pred_b),
    )


def aggregate(scores: dict[str, DocScore]) -> dict[str, float]:
    """Feet-weighted mean of each metric over documents (counts are summed)."""
    if not scores:
        return {}
    w = np.array([s.labelled_ft for s in scores.values()])
    out: dict[str, float] = {"n_docs": float(len(scores)), "labelled_ft": float(w.sum())}
    for field in (
        "lithology_accuracy",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "depth_mae_ft",
        "cluster_ari",
    ):
        v = np.array([getattr(s, field) for s in scores.values()], dtype=float)
        ok = np.isfinite(v)
        out[field] = float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")
    return out


def lofo_folds(doc_ids: list[str], family_of) -> list[tuple[str, list[str], list[str]]]:
    """Leave-one-family-out folds: (held-out family, train doc ids, test doc ids)."""
    fam: dict[str, list[str]] = defaultdict(list)
    for d in doc_ids:
        fam[family_of(d)].append(d)
    return [(f, [d for d in doc_ids if family_of(d) != f], ids) for f, ids in sorted(fam.items())]


def majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]
