"""Score pipeline CSVs against hand-labelled ground truth.

Ground truth: outputs/gt/<log name>.csv with columns top_depth,base_depth,lithology
Predictions:  outputs/results/<log name>.csv (written by src.run)

python -m src.evaluate [--pred outputs/results] [--tol 2]
Metrics, over the depth range covered by the ground truth:
  litho_acc  share of depth (ft) where the predicted rock name matches
  bnd_f1     boundary F1, a predicted boundary counts if within +-tol ft of a true one
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SYNONYMS = {"clay": "shale", "shale": "shale", "claystone": "shale", "mudstone": "shale",
            "sand": "sandstone", "sand and sandstone": "sandstone", "sandstone": "sandstone"}


def norm(name):
    """Set of canonical names ('Clay/Shale' -> {'shale'}); Clay and Shale count as the same rock."""
    out = set()
    for part in str(name).split("/"):
        p = " ".join(part.lower().replace(".", "").split())
        out.add(SYNONYMS.get(p, p))
    return out


def read(path):
    with open(path) as f:
        return [(float(r["top_depth"]), float(r["base_depth"]), r["lithology"]) for r in csv.DictReader(f)]


def label_at(rows, d):
    for t, b, n in rows:
        if t <= d < b:
            return n
    return None


def score(gt, pred, tol=2.0, res=0.5):
    lo, hi = gt[0][0], gt[-1][1]
    depths = np.arange(lo, hi, res)
    ok = [bool(norm(label_at(gt, d)) & norm(label_at(pred, d) or "")) for d in depths]
    acc = float(np.mean(ok))
    tb = [t for t, _, _ in gt[1:]]
    pb = [t for t, _, _ in pred if lo + tol < t < hi - tol]
    used, tp = set(), 0
    for b in tb:
        cand = [i for i, p in enumerate(pb) if abs(p - b) <= tol and i not in used]
        if cand:
            used.add(min(cand, key=lambda i: abs(pb[i] - b)))
            tp += 1
    prec = tp / len(pb) if pb else (1.0 if not tb else 0.0)
    rec = tp / len(tb) if tb else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(litho_acc=acc, bnd_f1=f1, bnd_p=prec, bnd_r=rec, n_true=len(tb), n_pred=len(pb))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="outputs/gt")
    ap.add_argument("--pred", default="outputs/results")
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()
    scores = []
    for g in sorted(Path(a.gt).glob("*.csv")):
        gt = read(g)
        if not gt:
            continue
        p = Path(a.pred) / g.name
        if not p.exists():
            print(f"{g.stem[:45]:45s} no prediction")
            continue
        s = score(gt, read(p), a.tol)
        scores.append(s)
        print(f"{g.stem[:45]:45s} litho_acc {s['litho_acc']:.1%}  boundary F1 {s['bnd_f1']:.2f} "
              f"(P {s['bnd_p']:.2f} R {s['bnd_r']:.2f}, {s['n_pred']} pred / {s['n_true']} true)")
    if scores:
        print(f"\nMEAN over {len(scores)} logs: litho_acc {np.mean([s['litho_acc'] for s in scores]):.1%}  "
              f"boundary F1 {np.mean([s['bnd_f1'] for s in scores]):.2f}")
