"""Command-line entry point: ``lith run | eval | check-data``."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import yaml

from lithology import __version__
from lithology.config import load_config
from lithology.io import Document, discover

log = logging.getLogger("lithology")
REPO = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _docs_from(paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    for p in map(Path, paths):
        if p.is_dir():
            disc = discover(p)
            for skipped, why in disc.skipped:
                log.info("skip %s: %s", skipped.name, why)
            docs += disc.documents
        else:
            ann = p.parent.parent / "bbox_annotation" / f"{p.stem}.json"
            docs.append(Document(p.stem, p, ann if ann.exists() else None))
    return docs


def _run_one(args: tuple[Document, str | None, str, bool]) -> dict:
    """Worker: one document in a fresh process (accurate peak memory, crash isolation)."""
    doc, config_path, out_dir, images = args
    import os

    os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # several Tesseract processes run at once: no oversubscription
    from lithology.pipeline import process, write_outputs

    cfg = load_config(Path(config_path) if config_path else None)
    t0 = time.perf_counter()
    res = process(doc, cfg)
    if res.ok:
        try:
            write_outputs(doc, res, Path(out_dir), cfg, image=images)
        except Exception as e:  # an output failure is reported, not fatal
            res.warnings.append(f"output failed: {type(e).__name__}: {e}")
    from lithology.pipeline import peak_rss_mb

    res.peak_rss_mb = round(peak_rss_mb(), 1)  # after the annotated image too: the true peak of this document
    return {
        "doc_id": doc.doc_id,
        "ok": res.ok,
        "error": res.error or "",
        "n_intervals": len(res.intervals),
        "n_legend": len(res.legend),
        "px_per_ft": round(res.scale.px_per_ft, 3) if res.scale else "",
        "scale_points": res.scale.n_points if res.scale else 0,
        "scale_conf": res.scale.confidence if res.scale else 0,
        "seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": res.peak_rss_mb,
        "timings": json.dumps(res.timings_s),
        "warnings": " | ".join(res.warnings),
        "_needs_review": res.layout.get("needs_review", []),  # run_meta.json only, not a run_summary.csv column
    }


# Measured 2026-11-07 on all 51 logs: peak memory <= decoded image + ~1.8 GB (OCR, pretrained model, crops).
_BASE_MB = 1800.0


def estimate_peak_mb(doc: Document) -> float:
    """Peak memory estimate for one document, from the image header only (nothing is decoded)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(doc.image_path) as im:
            w, h = im.size
            bpp = 1 if im.mode in ("1", "L", "P") else 2 if im.mode.startswith("I;16") else 4
    except Exception:
        return 2 * _BASE_MB
    return w * h * bpp / 1e6 + _BASE_MB


def _error_row(doc: Document, error: str) -> dict:
    return {
        "doc_id": doc.doc_id,
        "ok": False,
        "error": error,
        "n_intervals": 0,
        "n_legend": 0,
        "px_per_ft": "",
        "scale_points": 0,
        "scale_conf": 0,
        "seconds": 0.0,
        "peak_rss_mb": 0.0,
        "timings": "{}",
        "warnings": "",
    }


def run_batch(
    jobs: list[tuple], est_mb: list[float], workers: int, budget_mb: float, on_row=None, fn=None
) -> list[dict]:
    """Run documents in worker processes; a worker that dies (e.g. out of memory) never stops the batch.

    Jobs are started largest first, and only while the estimated peak memory of the running ones fits the budget
    (a job larger than the whole budget runs alone). A dead worker breaks the whole pool, so every job running at
    that moment is retried later, one at a time; a job that kills its worker even when alone is reported as an error.
    """
    fn = fn or _run_one
    queue = deque(sorted(range(len(jobs)), key=lambda i: -est_mb[i]))
    rows: dict[int, dict] = {}
    suspects: list[int] = []

    def emit(i: int, row: dict) -> None:
        rows[i] = row
        if on_row is not None:
            on_row(row)

    def drain(order: deque, n_workers: int, budget: float, retry: bool) -> None:
        running: dict[Future, int] = {}
        ex = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1)
        try:
            while order or running:
                used = sum(est_mb[i] for i in running.values())
                while order and len(running) < n_workers:
                    fit = next((i for i in order if used + est_mb[i] <= budget), None)
                    if fit is None:
                        if running:
                            break
                        fit = order[0]  # too big for the budget: run it alone
                    order.remove(fit)
                    running[ex.submit(fn, jobs[fit])] = fit
                    used += est_mb[fit]
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                broken = False
                for f in done:
                    i = running.pop(f)
                    try:
                        emit(i, f.result())
                    except BrokenProcessPool:
                        broken = True
                        if retry:
                            emit(i, _error_row(jobs[i][0], "worker process died (likely out of memory)"))
                        else:
                            suspects.append(i)
                    except Exception as e:  # anything else from the worker is that document's error
                        emit(i, _error_row(jobs[i][0], f"{type(e).__name__}: {e}"))
                if broken:  # the pool is unusable: everything still running is lost with it
                    for i in running.values():
                        if retry:
                            order.appendleft(i)
                        else:
                            suspects.append(i)
                    running.clear()
                    ex.shutdown(wait=False, cancel_futures=True)
                    ex = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    drain(queue, workers, budget_mb, retry=False)
    if suspects:
        log.warning("%d document(s) lost with a dead worker: retrying them one at a time", len(suspects))
        drain(deque(suspects), 1, float("inf"), retry=True)
    return [rows[i] for i in range(len(jobs))]


def tier2_problem(cfg) -> str | None:
    """Why the configured tier-2 second opinion cannot run (None when it can). Naming accuracy drops without it,
    so a run refuses to start instead of silently degrading."""
    if cfg.ai.tier < 2:
        return None
    from lithology.ai.embed import availability

    reason = availability()
    if reason is None:
        return None
    return (
        f"tier-2 naming model unavailable: {reason}. Install it (`pip install -e '.[ai]'` then `lith fetch-weights`) "
        "or run classical-only on purpose with `--tier 1`"
    )


def cmd_fetch_weights(a: argparse.Namespace) -> int:
    from lithology.ai.embed import availability, fetch_weights

    fetch_weights()
    reason = availability()
    if reason:
        log.error("weights still unavailable: %s", reason)
        return 1
    log.info("tier-2 weights cached; runs are offline from now on")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    overrides: dict = {}
    if a.detect_boxes:  # ignore the annotation files: the boxes come from the detector (bonus)
        overrides.setdefault("layout", {})["source"] = "detect"
    if a.tier is not None:
        overrides.setdefault("ai", {})["tier"] = a.tier
    if overrides:  # workers load the config from a path: write the override beside the output
        base = (yaml.safe_load(Path(a.config).read_text()) if a.config else {}) or {}
        for section, vals in overrides.items():
            base.setdefault(section, {}).update(vals)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        forced = Path(a.out) / "_config_run_overrides.yaml"
        forced.write_text(yaml.safe_dump(base))
        a.config = str(forced)
    cfg = load_config(Path(a.config) if a.config else None)
    problem = tier2_problem(cfg)
    if problem:
        log.error("%s", problem)
        return 2
    docs = _docs_from(a.inputs)
    if not docs:
        log.error("no documents found in %s", a.inputs)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = a.workers or cfg.runtime.workers
    log.info(
        "running %d documents with %d workers (config %s, commit %s)",
        len(docs),
        workers,
        cfg.hash(),
        _git_commit(),
    )
    t0 = time.perf_counter()
    jobs = [(d, a.config, str(out), not a.no_images) for d in docs]

    def report(row: dict) -> None:
        status = "ok " if row["ok"] else "ERR"
        log.info(
            "%s %6.1fs %6.0f MB %4d intervals  %s %s",
            status,
            row["seconds"],
            row["peak_rss_mb"],
            row["n_intervals"],
            row["doc_id"][:70],
            row["error"],
        )

    est = [estimate_peak_mb(d) for d in docs]
    rows = run_batch(jobs, est, workers, cfg.runtime.memory_budget_mb, on_row=report)
    wall = time.perf_counter() - t0
    review = {r["doc_id"]: r.pop("_needs_review") for r in rows if "_needs_review" in r}
    review = {k: v for k, v in review.items() if v}
    if review:
        log.warning(
            "%d document(s) have detected layout boxes that need review (confidence < %s): %s",
            len(review),
            cfg.layout.review_threshold,
            "; ".join(f"{k} [{', '.join(v)}]" for k, v in review.items()),
        )
    with open(out / "run_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    secs = [r["seconds"] for r in rows]
    mem = [r["peak_rss_mb"] for r in rows]
    n_err = sum(not r["ok"] for r in rows)
    meta = {
        "version": __version__,
        "commit": _git_commit(),
        "config_hash": cfg.hash(),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "documents": len(rows),
        "errors": n_err,
        "wall_seconds": round(wall, 1),
        "mean_seconds_per_doc": round(sum(secs) / len(secs), 2),
        "max_seconds_per_doc": max(secs),
        "max_peak_rss_mb": max(mem),
        "workers": workers,
        "memory_budget_mb": cfg.runtime.memory_budget_mb,
        "layout_review_threshold": cfg.layout.review_threshold,
        "layout_needs_review": review,  # doc_id -> detected boxes below the threshold (empty: none, or annotated)
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "done: %d docs, %d errors, %.1fs wall, %.2fs/doc mean, peak %.0f MB",
        len(rows),
        n_err,
        wall,
        meta["mean_seconds_per_doc"],
        meta["max_peak_rss_mb"],
    )
    return 1 if n_err else 0


def _labels_manifest(labels_dir: Path) -> dict[str, str]:
    m = labels_dir / "manifest.yaml"
    if not m.exists():
        return {}
    data = yaml.safe_load(m.read_text()) or {}
    return {str(k): str(v) for k, v in (data.get("split") or {}).items()}


def cmd_eval(a: argparse.Namespace) -> int:
    from lithology.evaluate import aggregate, score_document
    from lithology.render import read_csv

    labels_dir, pred_dir = Path(a.labels), Path(a.pred)
    split = _labels_manifest(labels_dir)
    label_files = sorted(p for p in labels_dir.glob("*.csv"))
    if a.split != "all":
        label_files = [p for p in label_files if split.get(p.stem, "dev") == a.split]
    if not label_files:
        log.error("no label files for split '%s' in %s", a.split, labels_dir)
        return 2
    scores = {}
    for lf in label_files:
        pf = pred_dir / lf.name
        if not pf.exists():
            log.warning("missing prediction for %s", lf.stem)
            continue
        scores[lf.stem] = score_document(read_csv(pf), read_csv(lf), tolerance_ft=a.tolerance_ft)
        s = scores[lf.stem]
        log.info(
            "%-60s acc %.3f  bF1 %.3f  depthMAE %.2f ft  ARI %.3f",
            lf.stem[:60],
            s.lithology_accuracy,
            s.boundary_f1,
            s.depth_mae_ft,
            s.cluster_ari,
        )
    agg = aggregate(scores)
    log.info("AGGREGATE (%s): %s", a.split, {k: round(v, 3) for k, v in agg.items()})
    cfg = load_config(Path(a.config) if a.config else None)
    manifest = REPO / "data" / "manifest.json"
    data_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12] if manifest.exists() else "none"
    log_path = REPO / "eval" / "results_log.csv"
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "config_hash": cfg.hash(),
        "data_hash": data_hash,
        "split": a.split,
        "note": a.note or "",
        **{k: round(v, 4) for k, v in agg.items()},
    }
    new = not log_path.exists()
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    """Score a run against the eye-audit spot labels (labels/audit) and append the result to eval/audit_log.csv."""
    from lithology.evaluate import AuditScore, score_audit

    pred_dir, audit_dir = Path(a.pred), Path(a.audit)
    totals = AuditScore(0, 0, 0, 0)
    for af in sorted(audit_dir.glob("*.csv")):
        pf = pred_dir / "diagnostics" / af.name
        if not pf.exists():
            log.warning("missing prediction for %s", af.stem)
            continue
        with open(af, newline="") as f:
            audit = list(csv.DictReader(f))
        with open(pf, newline="") as f:
            pred = list(csv.DictReader(f))
        s = score_audit(pred, audit)
        keys = ("n", "correct", "flagged", "confident_wrong")
        totals = AuditScore(*(getattr(totals, k) + getattr(s, k) for k in keys))
        log.info(
            "%-55s %2d/%2d right  %2d flagged  %2d confident-wrong",
            af.stem[:55],
            s.correct,
            s.n,
            s.flagged,
            s.confident_wrong,
        )
    log.info(
        "AUDIT: %d/%d right (%.1f%%), %d flagged, %d confident-wrong",
        totals.correct,
        totals.n,
        100 * totals.accuracy,
        totals.flagged,
        totals.confident_wrong,
    )
    log_path = REPO / "eval" / "audit_log.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "timestamp",
                    "commit",
                    "config_hash",
                    "pred",
                    "note",
                    "n",
                    "correct",
                    "accuracy",
                    "flagged",
                    "confident_wrong",
                ]
            )
        cfg = load_config(Path(a.config) if a.config else None)
        w.writerow(
            [
                dt.datetime.now().isoformat(timespec="seconds"),
                _git_commit(),
                cfg.hash(),
                pred_dir.name,
                a.note or "",
                totals.n,
                totals.correct,
                round(totals.accuracy, 4),
                totals.flagged,
                totals.confident_wrong,
            ]
        )
    return 0


def cmd_check_data(a: argparse.Namespace) -> int:
    from PIL import Image

    disc = discover(Path(a.images))
    problems = 0
    files = {}
    for d in disc.documents:
        ann = d.annotation()
        if ann is None:
            log.warning("no annotation: %s", d.doc_id)
            problems += 1
        else:
            with Image.open(d.image_path) as im:
                if im.size != (ann.width, ann.height):
                    log.warning(
                        "size mismatch %s: image %s vs annotation %s",
                        d.doc_id,
                        im.size,
                        (ann.width, ann.height),
                    )
                    problems += 1
            for name, b in (
                ("depth", ann.boxes.depth_column),
                ("lithology", ann.boxes.lithology_column),
                ("legend", ann.boxes.legend),
            ):
                if b.x1 > ann.width or b.y1 > ann.height:
                    log.warning("%s box outside image in %s", name, d.doc_id)
                    problems += 1
        h = hashlib.sha256()
        with open(d.image_path, "rb") as f:
            while block := f.read(1 << 22):
                h.update(block)
        files[d.image_path.name] = h.hexdigest()
    for p, why in disc.skipped:
        log.info("skipped %s: %s", p.name, why)
    out = REPO / "data" / "manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"documents": len(disc.documents), "skipped": [p.name for p, _ in disc.skipped], "sha256": files},
            indent=2,
            sort_keys=True,
        )
    )
    log.info(
        "%d documents, %d skipped, %d problems; manifest written to %s",
        len(disc.documents),
        len(disc.skipped),
        problems,
        out.relative_to(REPO),
    )
    return 1 if problems else 0


def cmd_labels_make(a: argparse.Namespace) -> int:
    from lithology.labelkit import is_edited, make_item, save_manifest

    cfg = load_config(Path(a.config) if a.config else None)
    sel = yaml.safe_load(Path(a.selection).read_text())["documents"]
    docs = discover(Path(a.images)).documents
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    existing = {}
    if (out / "kit.json").exists():
        from lithology.labelkit import KitItem

        existing = {d["key"]: KitItem(**d) for d in json.loads((out / "kit.json").read_text())}
    if only and not only <= {s["key"] for s in sel}:
        log.error("unknown keys in --only: %s", sorted(only - {s["key"] for s in sel}))
        return 2
    items = []
    for s in sel:
        if only and s["key"] not in only:
            if s["key"] in existing:
                items.append(existing[s["key"]])  # untouched: keep the person's files and window
            continue
        doc = next((d for d in docs if d.doc_id.startswith(s["prefix"])), None)
        if doc is None:
            log.error("no document matches prefix %r", s["prefix"])
            return 2
        old = existing.get(s["key"])
        if old and is_edited(old, out) and not a.force:
            log.error(
                "%s.csv has edits (or predates edit tracking); not overwriting it. Use --force to replace.", s["key"]
            )
            items.append(old)
            continue
        window = (old.window_top_ft, old.window_bottom_ft) if old else None  # keep a window already shown
        item = make_item(doc, s["key"], s["split"], cfg, out, window=window)
        log.info(
            "%-12s %-8s %6.0f-%6.0f ft  %s",
            item.key,
            item.split,
            item.window_top_ft,
            item.window_bottom_ft,
            doc.doc_id[:60],
        )
        items.append(item)
    save_manifest(items, out)
    log.info("kit written to %s", out)
    return 0


def cmd_labels_import(a: argparse.Namespace) -> int:
    from lithology.labelkit import import_kit

    problems = import_kit(Path(a.kit), Path(a.labels))
    for p in problems:
        log.warning(p)
    log.info("imported labels into %s (%d problems)", a.labels, len(problems))
    return 1 if problems else 0


def cmd_dash_build(a: argparse.Namespace) -> int:
    from lithology.dash.paths import Paths
    from lithology.dash.store import build

    cfg = load_config(Path(a.config) if a.config else None)
    t0 = time.perf_counter()
    meta = build(
        Paths(),
        include_holdout=a.all,
        tiles=not a.no_tiles,
        rebuild_tiles=a.rebuild_tiles,
        workers=a.workers,
        unresolved_label=cfg.naming.unresolved_label,
    )
    log.info(
        "dashboard data: %d logs (%s) in %.1fs",
        meta["logs"],
        "holdout included" if a.all else f"{meta['holdout_hidden']} holdout logs hidden",
        time.perf_counter() - t0,
    )
    return 0 if meta["logs"] else 1


def cmd_dash(a: argparse.Namespace) -> int:
    import uvicorn

    from lithology.dash.paths import Paths

    if not Paths().db.exists():
        log.error("no dashboard data yet: run `lith dash-build` first")
        return 2
    if not a.dev and not (Paths().dist / "index.html").exists():
        log.error("front end not built: run `cd dashboard && npm install && npm run build` (or use --dev)")
        return 2
    url = f"http://{a.host}:{a.port}"
    log.info("dashboard on %s", url if not a.dev else f"{url} (API only; open the Vite dev server)")
    if a.open and not a.dev:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        "lithology.dash.app:create_app", factory=True, host=a.host, port=a.port, reload=a.dev, log_level="warning"
    )
    return 0


def _vendor_of(doc_id: str, vendors: dict) -> str:
    for k, v in vendors.items():
        if doc_id in v["documents"]:
            return k
    return "new:" + doc_id[:24]


def _review_evidence(rows: list[dict], threshold: float) -> dict:
    """Pages and boxes flagged "needs review", and (on annotated pages) whether the rule catches the wrong boxes:
    flagged boxes that were wrong (IoU < 0.5) vs right, and unflagged boxes that were wrong anyway."""
    from lithology.models import BOX_KEYS

    out: dict = {
        "threshold": threshold,
        "pages": sum(bool(r["needs_review"]) for r in rows),
        "boxes": sum(len(r["needs_review"].split(";")) for r in rows if r["needs_review"]),
    }
    for split in ("dev", "holdout", "all"):
        c = {"pages": 0, "flagged_wrong": 0, "flagged_right": 0, "unflagged_wrong": 0, "unflagged_right": 0}
        for r in rows:
            if (split != "all" and r["split"] != split) or r["iou_depth_column"] == "":
                continue
            c["pages"] += 1
            flagged = set(r["needs_review"].split(";")) if r["needs_review"] else set()
            for k in BOX_KEYS:
                wrong = float(r[f"iou_{k}"]) < 0.5
                c[("flagged_" if k in flagged else "unflagged_") + ("wrong" if wrong else "right")] += 1
        if c["pages"]:
            out[split] = c
    return out


def cmd_layout(a: argparse.Namespace) -> int:
    """Bonus: detect the three boxes without annotations; score against the annotations where they exist."""
    import os

    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    import numpy as np
    from PIL import ImageDraw

    from lithology.io import crop_rgb, open_image
    from lithology.layout import detect_boxes
    from lithology.models import BOX_KEYS, Box, needs_review

    cfg = load_config(Path(a.config) if a.config else None)
    thr = cfg.layout.review_threshold
    docs = _docs_from(a.inputs)
    if a.only:
        docs = [d for d in docs if any(o in d.doc_id for o in a.only.split(","))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vend = (yaml.safe_load((REPO / "eval" / "layout_vendors.yaml").read_text()) or {}).get("vendors", {})
    sel = yaml.safe_load((REPO / "labels" / "selection.yaml").read_text()) or {}
    holdout = [d["prefix"] for d in sel.get("documents", []) if d.get("split") == "holdout"]
    rows = []
    for d in docs:
        img = open_image(d.image_path)
        t0 = time.perf_counter()
        try:
            lay = detect_boxes(img, cfg.layout)
        except Exception as e:  # one page must not stop the batch
            log.error("layout failed on %s: %s", d.doc_id, e)
            continue
        secs = time.perf_counter() - t0
        pred = lay.boxes.to_dict()
        split = "holdout" if any(d.doc_id.startswith(h) for h in holdout) else "dev"
        row = {"doc_id": d.doc_id, "vendor": _vendor_of(d.doc_id, vend), "split": split, "seconds": round(secs, 1)}
        ann = d.annotation()
        cand_key = {
            "depth_column": "depth_all",
            "interpreted_lithology_column": "lith_all",
            "lithology_legend": "legend_all",
        }
        for k in BOX_KEYS:
            row[f"conf_{k}"] = lay.confidence[k]
            if ann:
                gt = Box.from_seq(ann.boxes.to_dict()[k])
                row[f"iou_{k}"] = round(Box.from_seq(pred[k]).iou(gt), 3)
                cands = [c for c in lay.diagnostics.get(cand_key[k], [])[:3] if c[2] > c[0] and c[3] > c[1]]
                row[f"oracle3_{k}"] = round(max([Box.from_seq(c).iou(gt) for c in cands] + [row[f"iou_{k}"]]), 3)
            else:
                row[f"iou_{k}"] = row[f"oracle3_{k}"] = ""
        review = needs_review(lay.confidence, thr)
        row["needs_review"] = ";".join(review)  # boxes a human must check (confidence < layout.review_threshold)
        rows.append(row)
        record = {
            "boxes": pred,
            "confidence": lay.confidence,
            "diagnostics": lay.diagnostics,
            **row,
            "needs_review": review,
            "review_threshold": thr,
        }
        (out / f"{d.doc_id}_layout.json").write_text(json.dumps(record, indent=1, default=str))
        if not a.no_images:  # page preview: detected boxes (thick), annotated boxes (thin black)
            W, H = img.size
            s = min(1.0, 3000.0 / H, 1600.0 / W)
            prev = crop_rgb(img, Box(0, 0, W, H), scale=s)
            pil = __import__("PIL.Image", fromlist=["Image"]).fromarray(prev)
            dr = ImageDraw.Draw(pil)
            col = {
                "depth_column": (220, 0, 0),
                "interpreted_lithology_column": (0, 110, 255),
                "lithology_legend": (255, 140, 0),
            }
            for k in BOX_KEYS:
                if ann:
                    dr.rectangle([v * s for v in ann.boxes.to_dict()[k]], outline=(0, 0, 0), width=1)
                dr.rectangle([v * s for v in pred[k]], outline=col[k], width=3)
            pil.save(out / f"{d.doc_id}_layout.png")
            # header-only preview at readable resolution (legend + track headers)
            top = int(lay.diagnostics.get("body", [H, H])[0])
            hb = Box(0, 0, W, max(2, min(H, top + int(0.5 * (lay.diagnostics.get("dpi") or 200)))))
            hs = min(1.0, 1600.0 / W)
            hp = __import__("PIL.Image", fromlist=["Image"]).fromarray(crop_rgb(img, hb, scale=hs))
            hd = ImageDraw.Draw(hp)
            for k in BOX_KEYS:
                if ann:
                    hd.rectangle([v * hs for v in ann.boxes.to_dict()[k]], outline=(0, 0, 0), width=1)
                hd.rectangle([v * hs for v in pred[k]], outline=col[k], width=3)
            hp.save(out / f"{d.doc_id}_layout_header.png")
        log.info(
            "%5.1fs  depth %s  lith %s  legend %s  %s",
            secs,
            row["iou_depth_column"],
            row["iou_interpreted_lithology_column"],
            row["iou_lithology_legend"],
            d.doc_id[:60],
        )
    if not rows:
        return 2
    with open(out / "layout_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "documents": len(rows),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 1),
        "max_seconds": max(r["seconds"] for r in rows),
    }
    for split in ("dev", "holdout"):
        rs = [r for r in rows if r["split"] == split and r["iou_depth_column"] != ""]
        if not rs:
            continue
        part = {"pages": len(rs)}
        for k in BOX_KEYS:
            v = np.array([r[f"iou_{k}"] for r in rs], float)
            by: dict[str, list[float]] = {}
            for r in rs:
                by.setdefault(r["vendor"], []).append(r[f"iou_{k}"])
            orc = np.array([r[f"oracle3_{k}"] for r in rs], float)
            conf = np.array([r[f"conf_{k}"] for r in rs], float)
            good = v > 0.5
            pos, neg = conf[good], conf[~good]
            auroc = (
                float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])) if len(pos) and len(neg) else None
            )
            part[k] = {
                "page_mean": round(float(v.mean()), 3),
                "vendor_macro": round(float(np.mean([np.mean(x) for x in by.values()])), 3),
                "iou_gt_0.5": int((v > 0.5).sum()),
                "iou_lt_0.3": int((v < 0.3).sum()),
                "oracle3_recall": round(float((orc > 0.5).mean()), 3),
                "selection_acc": round(float(good[orc > 0.5].mean()), 3) if (orc > 0.5).any() else None,
                "conf_auroc": round(auroc, 3) if auroc is not None else None,
            }
        summary[split] = part
    summary["needs_review"] = _review_evidence(rows, thr)
    (out / "layout_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("layout summary: %s", json.dumps(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lith", description="Lithology interval extraction from well-log scans")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process images and write CSVs + annotated images")
    r.add_argument("inputs", nargs="+", help="image files or folders")
    r.add_argument("--out", default="out")
    r.add_argument("--config")
    r.add_argument("--workers", type=int)
    r.add_argument("--no-images", action="store_true", help="skip annotated images (faster)")
    r.add_argument(
        "--detect-boxes",
        action="store_true",
        help="ignore annotation files and detect the three boxes (bonus; for end-to-end evaluation). "
        "Without it the annotation file is used when there is one and the boxes are detected when there is not",
    )
    r.add_argument(
        "--tier",
        type=int,
        choices=(1, 2),
        help="1 = classical naming only; 2 = + local pretrained second opinion (default from config; needs torch "
        "and cached weights: `lith fetch-weights`)",
    )
    r.set_defaults(fn=cmd_run)

    fw = sub.add_parser("fetch-weights", help="download the tier-2 model weights once, so later runs stay offline")
    fw.set_defaults(fn=cmd_fetch_weights)

    lo = sub.add_parser("layout", help="bonus: detect the three boxes without annotations, score against them")
    lo.add_argument("inputs", nargs="+", help="image files or folders")
    lo.add_argument("--out", default="out/layout")
    lo.add_argument("--config")
    lo.add_argument("--only", help="comma-separated substrings of document names")
    lo.add_argument("--no-images", action="store_true", help="skip the page previews")
    lo.set_defaults(fn=cmd_layout)

    e = sub.add_parser("eval", help="score predictions against hand labels")
    e.add_argument("--pred", default="out")
    e.add_argument("--labels", default="labels")
    e.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    e.add_argument("--tolerance-ft", type=float, default=2.0)
    e.add_argument("--config")
    e.add_argument("--note")
    e.set_defaults(fn=cmd_eval)

    au = sub.add_parser("audit", help="score a run against the eye-audit spot labels (labels/audit)")
    au.add_argument("--pred", default="out")
    au.add_argument("--audit", default=str(REPO / "labels" / "audit"))
    au.add_argument("--config")
    au.add_argument("--note")
    au.set_defaults(fn=cmd_audit)

    c = sub.add_parser("check-data", help="audit the dataset and write data/manifest.json")
    c.add_argument("--images", default="dataset/images")
    c.set_defaults(fn=cmd_check_data)

    lm = sub.add_parser("labels-make", help="build the hand-labelling kit (pictures + draft CSVs)")
    lm.add_argument("--selection", default="labels/selection.yaml")
    lm.add_argument("--images", default="dataset/images")
    lm.add_argument("--out", default="labels/kit")
    lm.add_argument("--config")
    lm.add_argument("--only", help="comma-separated kit keys to rebuild (others are left untouched)")
    lm.add_argument("--force", action="store_true", help="overwrite draft CSVs even if they were edited")
    lm.set_defaults(fn=cmd_labels_make)

    li = sub.add_parser("labels-import", help="turn corrected kit CSVs into label files")
    li.add_argument("--kit", default="labels/kit")
    li.add_argument("--labels", default="labels")
    li.set_defaults(fn=cmd_labels_import)

    db = sub.add_parser("dash-build", help="build the dashboard database and scan tiles from out/")
    db.add_argument("--all", action="store_true", help="include holdout logs (for the final demo only)")
    db.add_argument("--no-tiles", action="store_true", help="skip scan tiles (data only)")
    db.add_argument("--rebuild-tiles", action="store_true", help="re-cut tiles that already exist")
    db.add_argument("--workers", type=int, default=2)
    db.add_argument("--config")
    db.set_defaults(fn=cmd_dash_build)

    ds = sub.add_parser("dash", help="serve the dashboard")
    ds.add_argument("--host", default="127.0.0.1")
    ds.add_argument("--port", type=int, default=8000)
    ds.add_argument("--dev", action="store_true", help="API only with auto-reload (front end via `npm run dev`)")
    ds.add_argument("--open", action="store_true", help="open the browser")
    ds.set_defaults(fn=cmd_dash)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
