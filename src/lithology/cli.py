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
from concurrent.futures import ProcessPoolExecutor
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
    from lithology.pipeline import process, write_outputs

    cfg = load_config(Path(config_path) if config_path else None)
    t0 = time.perf_counter()
    res = process(doc, cfg)
    if res.ok:
        try:
            write_outputs(doc, res, Path(out_dir), cfg, image=images)
        except Exception as e:  # an output failure is reported, not fatal
            res.warnings.append(f"output failed: {type(e).__name__}: {e}")
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
    }


def cmd_run(a: argparse.Namespace) -> int:
    cfg = load_config(Path(a.config) if a.config else None)
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
    rows = []
    jobs = [(d, a.config, str(out), not a.no_images) for d in docs]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        for row in ex.map(_run_one, jobs):
            rows.append(row)
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
    wall = time.perf_counter() - t0
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
    from lithology.labelkit import make_item, save_manifest

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
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="score predictions against hand labels")
    e.add_argument("--pred", default="out")
    e.add_argument("--labels", default="labels")
    e.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    e.add_argument("--tolerance-ft", type=float, default=2.0)
    e.add_argument("--config")
    e.add_argument("--note")
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("check-data", help="audit the dataset and write data/manifest.json")
    c.add_argument("--images", default="dataset/images")
    c.set_defaults(fn=cmd_check_data)

    lm = sub.add_parser("labels-make", help="build the hand-labelling kit (pictures + draft CSVs)")
    lm.add_argument("--selection", default="labels/selection.yaml")
    lm.add_argument("--images", default="dataset/images")
    lm.add_argument("--out", default="labels/kit")
    lm.add_argument("--config")
    lm.add_argument("--only", help="comma-separated kit keys to rebuild (others are left untouched)")
    lm.set_defaults(fn=cmd_labels_make)

    li = sub.add_parser("labels-import", help="turn corrected kit CSVs into label files")
    li.add_argument("--kit", default="labels/kit")
    li.add_argument("--labels", default="labels")
    li.set_defaults(fn=cmd_labels_import)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
