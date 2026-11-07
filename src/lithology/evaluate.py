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


def lenient_match(pred: str, truth: str) -> bool:
    """Same rock when every word of one name appears in the other ("Sand and Sandstone" ~ "Sandstone")."""
    p = set(re.findall(r"[a-z0-9]+", pred.lower()))
    t = set(re.findall(r"[a-z0-9]+", truth.lower()))
    return bool(p and t) and (p <= t or t <= p)


@dataclass(frozen=True)
class DocScore:
    labelled_ft: float
    lithology_accuracy: float  # share of labelled feet whose predicted lithology matches exactly
    lithology_accuracy_lenient: float  # same, counting "Sand and Sandstone" ~ "Sandstone" as a match
    unresolved_share: float  # share of labelled feet the pipeline declined to name
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
    raw_pred = _rows_to_labels(pred, y0, y1, lambda iv: iv.lithology)
    raw_truth = _rows_to_labels(label, y0, y1, lambda iv: iv.lithology)
    pairs = Counter(zip(raw_pred[labelled], raw_truth[labelled], strict=True))
    n_lab = int(labelled.sum()) or 1
    lenient = sum(n for (p, t), n in pairs.items() if p != UNLABELLED and lenient_match(p, t)) / n_lab
    unresolved = sum(n for (p, _), n in pairs.items() if normalise_name(p) == "unresolved") / n_lab

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
        lithology_accuracy_lenient=float(lenient),
        unresolved_share=float(unresolved),
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
        "lithology_accuracy_lenient",
        "unresolved_share",
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


# ------------------------------------------------------------------------------------------- eye audit
# Spot labels made by looking at the scan (labels/audit/<doc_id>.csv): short windows in the middle of beds, each
# with the rock a person (or Claude, source=claude-vision) reads from the column and legend. They are not complete
# interval labels, so only naming is scored: the predicted name covering most of each window.

FLAG_NOTE = "could not differentiate"


@dataclass(frozen=True)
class AuditScore:
    n: int
    correct: int  # predicted name equals the audited name (case/punctuation-insensitive)
    flagged: int  # predicted bed is Unresolved or carries the "could not differentiate" note
    confident_wrong: int  # wrong name presented as an answer (not flagged as a close call)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")


def score_audit(pred_rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> AuditScore:
    """``pred_rows``: diagnostics CSV rows (with confidence, notes). ``audit_rows``: audit CSV rows."""
    n = correct = flagged = confident_wrong = 0
    for a in audit_rows:
        y0, y1 = int(float(a["top_y"])), int(float(a["bottom_y"]))
        cover: Counter = Counter()
        info: dict[str, tuple[float, bool]] = {}
        for p in pred_rows:
            ov = min(y1, int(float(p["bottom_y"]))) - max(y0, int(float(p["top_y"])))
            if ov > 0:
                cover[p["lithology"]] += ov
                st = p.get("status")
                legacy = normalise_name(p["lithology"]) == "unresolved" or FLAG_NOTE in (p.get("notes") or "")
                flag = (st != "named") if st else legacy
                info[p["lithology"]] = (float(p.get("confidence") or 0), flag)
        n += 1
        if not cover:
            continue
        name = cover.most_common(1)[0][0]
        _, flag = info[name]
        ok = normalise_name(name) == normalise_name(a["lithology"])
        correct += ok
        flagged += flag
        confident_wrong += (not ok) and not flag
    return AuditScore(n, correct, flagged, confident_wrong)
