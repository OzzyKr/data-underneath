"""Layout page of the dashboard: step 1 (the box detector) on every image, scored against the annotations, and the
same detector run on an uploaded image. Reads the newest ``lith layout`` output under ``out/``."""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from lithology.dash.paths import Paths
from lithology.dash.titles import short_title
from lithology.io import crop_rgb, depth_reference, discover, image_dpi, open_image, scale_inches_per_100ft
from lithology.models import BOX_KEYS, Box

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_CROP_W = 1800
MAX_CROP_H = 6000


def latest_run(paths: Paths) -> Path | None:
    """The newest layout run under out/ (or ``LITH_LAYOUT_DIR``): a folder holding ``layout_summary.json``."""
    env = os.environ.get("LITH_LAYOUT_DIR")
    if env:
        p = Path(env)
        return p if (p / "layout_summary.json").exists() else None
    runs = [p for p in paths.out_batch.glob("layout*") if (p / "layout_summary.json").exists()]
    return max(runs, key=lambda p: (p / "layout_summary.json").stat().st_mtime) if runs else None


def _variant(doc_id: str) -> str:
    inches = scale_inches_per_100ft(doc_id)
    return " ".join(p for p in (depth_reference(doc_id), f"{inches:g} in" if inches else "") if p)


def _num(v: str) -> float | None:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


class LayoutView:
    """Everything the Layout page serves. Holdout pages stay hidden unless the dashboard was built with ``--all``."""

    def __init__(self, paths: Paths, include_holdout: bool):
        self.paths = paths
        self.include_holdout = include_holdout
        self.run = latest_run(paths)
        self.docs = {d.doc_id: d for d in discover(paths.images).documents}
        self.uploads_dir = paths.uploads / "layout"
        self.crops_dir = paths.cache / "layout_crops"
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        if self.run:
            for r in csv.DictReader((self.run / "layout_summary.csv").open()):
                if r["split"] == "holdout" and not include_holdout:
                    continue
                self.rows[r["doc_id"]] = {
                    "doc_id": r["doc_id"],
                    "title": short_title(r["doc_id"]),  # the same short name the other pages use
                    "variant": _variant(r["doc_id"]),  # tells apart scans of one well (MD / TVD, scale)
                    "vendor": r["vendor"],
                    "split": r["split"],
                    "seconds": _num(r["seconds"]),
                    "iou": {k: _num(r.get(f"iou_{k}", "")) for k in BOX_KEYS},
                    "confidence": {k: _num(r.get(f"conf_{k}", "")) for k in BOX_KEYS},
                }

    # ------------------------------------------------------------------ batch results

    def summary(self) -> dict[str, Any]:
        if not self.run:
            return {"run": None, "pages": [], "metrics": None}
        s = json.loads((self.run / "layout_summary.json").read_text())
        metrics = {"dev": s.get("dev")}
        if self.include_holdout:
            metrics["holdout"] = s.get("holdout")
        return {
            "run": self.run.name,
            "mean_seconds": s.get("mean_seconds"),
            "max_seconds": s.get("max_seconds"),
            "metrics": metrics,
            "pages": list(self.rows.values()),
            "uploads": [self._job_public(j) for j in self.jobs.values()],
        }

    def detail(self, doc_id: str) -> dict[str, Any] | None:
        if doc_id in self.jobs:
            return self._job_detail(self.jobs[doc_id])
        row = self.rows.get(doc_id)
        if row is None or not self.run:
            return None
        res = json.loads((self.run / f"{doc_id}_layout.json").read_text())
        doc = self.docs.get(doc_id)
        ann = doc.annotation() if doc else None
        img = open_image(doc.image_path) if doc else None
        diag = res.get("diagnostics", {})
        return {
            **row,
            "source": "batch",
            "size": list(img.size) if img else None,
            "dpi": diag.get("dpi") or (image_dpi(img) if img else None),
            "body": diag.get("body"),
            "boxes": res["boxes"],
            "annotated": ann.boxes.to_dict() if ann else None,
            "notes": _notes(diag),
        }

    # ------------------------------------------------------------------ uploads

    def submit(self, name: str, data: bytes) -> str:
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_EXT:
            raise ValueError(f"not an image: {name} (use .tif, .tiff, .png or .jpg)")
        try:  # reject unreadable files now, with a plain message, instead of failing in the detector
            with Image.open(io.BytesIO(data)) as probe:
                probe.size  # noqa: B018
        except Exception as e:
            raise ValueError(f"could not read {name} as an image") from e
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        job_id = "upload-" + uuid.uuid4().hex[:10]
        path = self.uploads_dir / f"{job_id}{ext}"
        path.write_bytes(data)
        job = {"id": job_id, "name": name, "path": path, "state": "queued", "t0": time.time(), "error": None}
        with self.lock:
            self.jobs[job_id] = job
        self.pool.submit(self._detect, job)
        return job_id

    def _detect(self, job: dict[str, Any]) -> None:
        from lithology.config import load_config
        from lithology.layout import detect_boxes

        job["state"] = "running"
        try:
            img = open_image(job["path"])
            t = time.perf_counter()
            res = detect_boxes(img, load_config().layout)
            job.update(
                state="done",
                seconds=round(time.perf_counter() - t, 1),
                size=list(img.size),
                dpi=res.diagnostics.get("dpi"),
                body=res.diagnostics.get("body"),
                boxes=res.boxes.to_dict(),
                confidence={k: round(float(v), 3) for k, v in res.confidence.items()},
                notes=_notes(res.diagnostics),
            )
        except Exception as e:  # shown to the user; the dashboard keeps running
            job.update(state="error", error=f"{type(e).__name__}: {e}")

    def job(self, job_id: str) -> dict[str, Any] | None:
        j = self.jobs.get(job_id)
        return self._job_public(j) if j else None

    @staticmethod
    def _job_public(j: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": j["id"],
            "name": j["name"],
            "state": j["state"],
            "error": j["error"],
            "elapsed": round(time.time() - j["t0"], 1),
            "seconds": j.get("seconds"),
            "confidence": j.get("confidence"),
        }

    def _job_detail(self, j: dict[str, Any]) -> dict[str, Any] | None:
        if j["state"] != "done":
            return None
        return {
            "doc_id": j["id"],
            "vendor": "upload",
            "split": "upload",
            "seconds": j["seconds"],
            "iou": None,
            "confidence": j["confidence"],
            "source": "upload",
            "name": j["name"],
            "size": j["size"],
            "dpi": j["dpi"],
            "body": j["body"],
            "boxes": j["boxes"],
            "annotated": None,
            "notes": j["notes"],
        }

    # ------------------------------------------------------------------ pixels

    def image_path(self, doc_id: str) -> Path | None:
        if doc_id in self.jobs:
            return self.jobs[doc_id]["path"]
        if doc_id in self.rows and doc_id in self.docs:
            return self.docs[doc_id].image_path
        return None

    def crop_png(self, doc_id: str, y0: int, y1: int, width: int) -> bytes | None:
        """PNG of the full-width band y0..y1, shrunk to at most ``width`` px wide (cached on disk)."""
        path = self.image_path(doc_id)
        if path is None:
            return None
        img = open_image(path)
        W, H = img.size
        y0, y1 = max(0, min(int(y0), H - 1)), max(1, min(int(y1), H))
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        width = max(64, min(int(width), MAX_CROP_W))
        scale = min(1.0, width / W, MAX_CROP_H / max(1, y1 - y0))
        key = f"{doc_id}_{y0}_{y1}_{round(scale * 1000)}.png".replace("/", "_")
        f = self.crops_dir / key
        if f.exists():
            return f.read_bytes()
        arr = crop_rgb(img, Box(0, y0, W, y1), scale=scale)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=False)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        f.write_bytes(buf.getvalue())
        return buf.getvalue()

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _notes(diag: dict[str, Any]) -> list[str]:
    """Short plain-language notes on how the detector decided, from its diagnostics."""
    out = []
    if diag.get("bw"):
        out.append("Black-and-white page: the lithology column was judged by hatch texture.")
    if diag.get("noise"):
        out.append(f"Noisy scan (noise {diag['noise']}): colour cues read on smoothed pixels.")
    anchors = diag.get("anchors") or []
    if any(a[2] for a in anchors if len(a) > 2):
        out.append('An "Interp Lith"-style header named the lithology column.')
    elif anchors:
        out.append('A "Lith" header word was found over a track.')
    if diag.get("neg_anchors"):
        out.append("Cuttings / % tracks were recognised and excluded.")
    rots = diag.get("depth_rotations")
    if rots:
        out.append(f"Depth labels read at rotation {', '.join(str(r) for r in rots)}°.")
    return out
