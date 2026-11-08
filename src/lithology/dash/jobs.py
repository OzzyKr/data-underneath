"""Live runs: a scan dropped on the dashboard goes through the same pipeline as ``lith run``.

Each run is a worker process that reports its steps to a status file; results land in ``out_live/``. Boxes come
from the uploaded box file when there is one, else from the layout detector (``boxes="detect"`` forces it).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from lithology.dash.paths import Paths
from lithology.models import Annotation

log = logging.getLogger(__name__)

STEPS = ["load", "layout", "scale", "segment", "cluster", "legend", "assemble", "write", "tiles"]
SUMMARY_FIELDS = [
    "doc_id", "ok", "error", "n_intervals", "n_legend", "px_per_ft", "scale_points", "scale_conf", "seconds",
    "peak_rss_mb", "timings", "warnings",
]  # fmt: skip
IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
BOX_MODES = ("auto", "detect")  # auto: the box file when uploaded, else the detector; detect: always the detector


def _write_status(path: Path, **kw: Any) -> None:
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cur.update(kw)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur), encoding="utf-8")
    tmp.replace(path)


def run_live(
    doc_id: str, image: str, ann: str | None, out_dir: str, status: str, tiles_dir: str, detect: bool = False
) -> dict[str, Any]:
    """Worker process: pipeline + outputs + tiles for one uploaded scan. Mirrors ``cli._run_one``.

    ``ann`` is the uploaded box file (None: the detector finds the boxes); ``detect`` forces the detector."""
    from lithology.config import load_config
    from lithology.dash.tiles import build_tiles
    from lithology.io import Document
    from lithology.pipeline import process, write_outputs

    st = Path(status)
    started: dict[str, float] = {}
    done: dict[str, float] = {}

    def on_step(name: str) -> None:
        now = time.perf_counter()
        for prev, t in started.items():
            done.setdefault(prev, round(now - t, 2))
        started[name] = now
        _write_status(st, state="running", step=name, steps=done)

    t0 = time.perf_counter()
    cfg = load_config(overrides={"layout": {"source": "detect"}} if detect else None)
    doc = Document(doc_id, Path(image), Path(ann) if ann else None)
    res = process(doc, cfg, on_step=on_step)
    for prev, t in started.items():
        done.setdefault(prev, round(time.perf_counter() - t, 2))
    if not res.ok:
        _write_status(st, state="error", error=res.error, steps=done)
        return {"ok": False, "error": res.error}
    pipeline_s = round(time.perf_counter() - t0, 2)
    boxes = _boxes_report(res.layout)

    _write_status(st, step="write", steps=done)
    t = time.perf_counter()
    out = Path(out_dir)
    write_outputs(doc, res, out, cfg, image=False)
    row = {
        "doc_id": doc_id,
        "ok": True,
        "error": "",
        "n_intervals": len(res.intervals),
        "n_legend": len(res.legend),
        "px_per_ft": round(res.scale.px_per_ft, 3) if res.scale else "",
        "scale_points": res.scale.n_points if res.scale else 0,
        "scale_conf": res.scale.confidence if res.scale else 0,
        "seconds": pipeline_s,
        "peak_rss_mb": res.peak_rss_mb,
        "timings": json.dumps(res.timings_s),
        "warnings": " | ".join(res.warnings),
    }
    summary = out / "run_summary.csv"
    rows = []
    if summary.exists():
        with open(summary, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["doc_id"] != doc_id]
    rows.append(row)
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    # the boxes the run actually used, in the organisers' format: the dashboard draws and tiles from this file
    with Image.open(image) as im:
        w, h = im.size
    used = Path(image).with_name(f"{doc_id}.boxes.json")
    used.write_text(
        json.dumps({"image_name": Path(image).name, "width": w, "height": h, "bounding_boxes": res.boxes.to_dict()}),
        encoding="utf-8",
    )
    reg = out / "sources.json"
    sources = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {}
    sources[doc_id] = {"image": image, "ann": str(used), "uploaded_ann": ann, "boxes_source": boxes["source"]}
    reg.write_text(json.dumps(sources, indent=1), encoding="utf-8")
    done["write"] = round(time.perf_counter() - t, 2)

    _write_status(st, step="tiles", steps=done)
    t = time.perf_counter()
    legend = json.loads((out / "diagnostics" / f"{doc_id}_legend.json").read_text(encoding="utf-8"))
    build_tiles(Path(image), res.boxes, legend, Path(tiles_dir))
    done["tiles"] = round(time.perf_counter() - t, 2)
    _write_status(
        st,
        state="ingesting",
        step="ingest",
        steps=done,
        pipeline_seconds=pipeline_s,
        peak_rss_mb=res.peak_rss_mb,
        boxes=boxes,
    )
    return {"ok": True, "row": {k: str(v) for k, v in row.items()}}


def _boxes_report(layout: dict[str, Any]) -> dict[str, Any]:
    """Where the run's boxes came from: "json" or "detector" (with confidence, and IoU vs any uploaded box file)."""
    if layout.get("source") != "detected":
        return {"source": "json"}
    return {
        "source": "detector",
        "confidence": layout.get("confidence", {}),
        "low_confidence": layout.get("low_confidence", []),
        "iou_vs_json": layout.get("iou_vs_annotation"),
    }


class Jobs:
    def __init__(self, paths: Paths, ingest: Callable[[str, dict[str, str]], str]):
        self.paths = paths
        self.ingest = ingest  # (doc_id, summary row) -> log_id, called in this process when a run finishes
        self.pool = ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def submit(self, files: list[tuple[str, bytes]], boxes: str = "auto") -> str:
        """Queue one scan.

        ``boxes``: "auto" uses the box file when uploaded, else the detector; "detect" always detects.
        """
        if boxes not in BOX_MODES:
            raise ValueError(f"boxes must be one of {', '.join(BOX_MODES)}")
        images = [(n, b) for n, b in files if Path(n).suffix.lower() in IMAGE_EXT]
        jsons = [(n, b) for n, b in files if Path(n).suffix.lower() == ".json"]
        if len(images) != 1:
            raise ValueError("drop exactly one scan (.tif, .tiff, .png or .jpg)")
        img_name, img_bytes = images[0]
        doc_id = Path(img_name).stem
        job_id = uuid.uuid4().hex[:8]
        folder = self.paths.uploads / job_id
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[/\\]", "_", Path(img_name).name)
        img_path = folder / safe
        img_path.write_bytes(img_bytes)
        try:
            with Image.open(img_path) as im:
                size = im.size
        except Exception as e:
            raise ValueError(f"cannot read the scan: {e}") from e
        ann_path: Path | None = None  # no box file: the layout detector finds the boxes
        if jsons:
            ann_path = folder / f"{doc_id}.json"
            ann_path.write_bytes(jsons[0][1])
            try:
                ann = Annotation.from_json(ann_path)
            except Exception as e:
                raise ValueError(f"the .json is not a bounding-box file: {e}") from e
            if size != (ann.width, ann.height):
                raise ValueError(
                    f"the boxes are for a {ann.width}x{ann.height} image but the scan is {size[0]}x{size[1]}"
                )

        from lithology.dash.store import slug

        status = folder / "status.json"
        _write_status(status, state="queued", step=None, steps={}, doc_id=doc_id, size=list(size))
        job = {"id": job_id, "doc_id": doc_id, "status": status, "started": time.time(), "log_id": None, "error": None}
        with self.lock:
            self.jobs[job_id] = job
        fut = self.pool.submit(
            run_live,
            doc_id,
            str(img_path),
            str(ann_path) if ann_path else None,
            str(self.paths.out_live),
            str(status),
            str(self.paths.tiles / slug("live", doc_id)),
            boxes == "detect",
        )
        fut.add_done_callback(lambda f: self._finished(job_id, f))
        return job_id

    def _finished(self, job_id: str, fut: Future) -> None:
        job = self.jobs[job_id]
        try:
            res = fut.result()
            if not res.get("ok"):
                job["error"] = res.get("error") or "pipeline failed"
                return
            job["log_id"] = self.ingest(job["doc_id"], res["row"])
            _write_status(job["status"], state="done", step=None)
        except Exception as e:
            log.exception("live run %s failed", job_id)
            job["error"] = f"{type(e).__name__}: {e}"
            _write_status(job["status"], state="error", error=job["error"])
        finally:
            job["finished"] = time.time()

    def status(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        st = json.loads(job["status"].read_text(encoding="utf-8")) if job["status"].exists() else {}
        state = "error" if job["error"] else st.get("state", "queued")
        end = job.get("finished") or time.time()
        return {
            "id": job_id,
            "doc_id": job["doc_id"],
            "state": state,
            "step": st.get("step"),
            "steps": st.get("steps", {}),
            "order": STEPS,
            "elapsed": round(end - job["started"], 1),
            "pipeline_seconds": st.get("pipeline_seconds"),
            "peak_rss_mb": st.get("peak_rss_mb"),
            "size": st.get("size"),
            "boxes": st.get("boxes"),
            "log_id": job["log_id"],
            "error": job["error"] or st.get("error"),
        }

    def shutdown(self) -> None:
        # wait=True: with max_tasks_per_child, a non-waiting shutdown races the pool's manager thread (Python 3.12)
        self.pool.shutdown(wait=True, cancel_futures=True)
