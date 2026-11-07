"""Run every step for one document, with timing, a memory reading and a guard around each step."""

from __future__ import annotations

import logging
import math
import resource
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lithology.checks import check_intervals
from lithology.cluster import cluster_intervals, merge_adjacent
from lithology.config import Config
from lithology.features import match_descriptor
from lithology.io import Document, crop_rgb, image_dpi, open_image
from lithology.legend import match_clusters, read_legend, swatch_descriptor
from lithology.models import Box, DocumentBoxes, DocumentResult, Interval
from lithology.render import render_annotated, write_csv
from lithology.scale import estimate_scale, prior_px_per_ft
from lithology.segment import core_box_x, interval_descriptors, segment_column

log = logging.getLogger(__name__)

DEFAULT_PX_PER_FT = 10.0  # only used for block sizes when neither OCR nor filename/DPI gives a scale


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


@contextmanager
def _timed(timings: dict[str, float], name: str) -> Generator[None, None, None]:
    t = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - t, 3)


def _cluster_match_descriptor(
    column: np.ndarray, filled: list, ids: np.ndarray, c: int, block: int, cfg: Config, max_rows: int = 3000
) -> np.ndarray:
    """Match descriptor of one cluster from its thickest intervals (trimmed away from their boundaries)."""
    members = sorted((s for s, i in zip(filled, ids, strict=True) if i == c), key=lambda s: s.top - s.bottom)
    parts, rows = [], 0
    for s in members:
        a, b = s.top + block, s.bottom - block
        if b - a < 4:
            a, b = s.top, s.bottom
        take = min(b - a, max_rows - rows)
        parts.append(column[a : a + take])
        rows += take
        if rows >= max_rows:
            break
    return match_descriptor(np.concatenate(parts), cfg.segment.ink_luma)


def process(doc: Document, cfg: Config, boxes: DocumentBoxes | None = None) -> DocumentResult:
    """Extract intervals for one document. Never raises: errors are recorded on the result."""
    res = DocumentResult(doc_id=doc.doc_id, intervals=[])
    try:
        _process(doc, cfg, boxes, res)
    except Exception as e:  # the batch must survive any single document
        res.error = f"{type(e).__name__}: {e}"
        log.error("%s failed: %s\n%s", doc.doc_id, res.error, traceback.format_exc())
    res.peak_rss_mb = round(peak_rss_mb(), 1)
    return res


def _process(doc: Document, cfg: Config, boxes: DocumentBoxes | None, res: DocumentResult) -> None:
    t = res.timings_s
    with _timed(t, "load"):
        img = open_image(doc.image_path)
        if boxes is None:
            ann = doc.annotation()
            if ann is None:
                raise ValueError("no boxes supplied and no annotation file (bonus detector not enabled)")
            boxes = ann.boxes
        res.boxes = boxes
        lc = boxes.lithology_column.clip(*img.size)

    with _timed(t, "scale"):
        scale, warns = estimate_scale(img, boxes.depth_column, doc.doc_id, cfg.scale)
        res.scale = scale
        res.warnings += warns
        px_per_ft = scale.px_per_ft if scale else (prior_px_per_ft(img, doc.doc_id) or DEFAULT_PX_PER_FT)

    with _timed(t, "segment"):
        xa, xb = core_box_x(lc.x0, lc.x1, cfg.segment.core_width_fraction)
        column = crop_rgb(img, Box(xa, lc.y0, xb, lc.y1))
        seg = segment_column(column, px_per_ft, cfg.segment)
        filled = [s for s in seg.segments if not s.blank]

    with _timed(t, "cluster"):
        desc = interval_descriptors(seg, filled) if filled else np.zeros((0, 12))
        ids = cluster_intervals(desc, seg.scaler, cfg.cluster)

    with _timed(t, "legend"):
        entries = []
        try:
            lg = boxes.legend.clip(*img.size)
            legend_rgb = crop_rgb(img, lg)
            entries = read_legend(legend_rgb, (lg.x0, lg.y0), image_dpi(img) or 200.0, cfg.legend)
        except Exception as e:  # a legend failure must not lose the intervals
            res.warnings.append(f"legend read failed: {type(e).__name__}: {e}")
        res.legend = entries
        names: dict[int, str] = {}
        match_conf: dict[int, float] = {}
        rocks = [e for e in entries if e.is_rock] or entries  # never name a bed after a drilling symbol
        if rocks and len(ids):
            entries = rocks
            sw = np.array([swatch_descriptor(crop_rgb(img, e.swatch), cfg.segment.ink_luma) for e in entries])
            n_clusters = int(ids.max()) + 1
            cdesc = np.array(
                [_cluster_match_descriptor(column, filled, ids, c, seg.block_px, cfg) for c in range(n_clusters)]
            )
            swatch_of: dict[int, int] = {}
            for c, m in enumerate(match_clusters(cdesc, sw)):
                swatch_of[c] = m.swatch
                names[c] = entries[m.swatch].label
                # certain when the winner is close and clearly ahead of the runner-up
                match_conf[c] = float(math.exp(-m.distance / 4.0) * (1 - math.exp(-m.margin)))
            # Groups the legend calls the same rock are one group: renumber by matched swatch (first appearance),
            # so cluster ids agree with names and touching intervals of the same rock merge.
            new_id: dict[int, int] = {}
            for sw_i in (swatch_of[int(c)] for c in ids):
                new_id.setdefault(sw_i, len(new_id))
            remap = {c: new_id[swatch_of[c]] for c in swatch_of}
            names = {remap[c]: n for c, n in names.items()}
            match_conf = {remap[c]: min(v for k, v in match_conf.items() if remap[k] == remap[c]) for c in match_conf}
            ids = np.array([remap[int(c)] for c in ids])
        else:
            res.warnings.append("no legend entries read; intervals named by cluster")
        del column

    with _timed(t, "assemble"):
        merged = merge_adjacent([s.top for s in filled], [s.bottom for s in filled], ids.tolist())
        intervals = []
        for top, bottom, c in merged:
            y0, y1 = lc.y0 + top, lc.y0 + bottom
            iv = Interval(
                top_y=y0,
                bottom_y=y1,
                cluster_id=int(c),
                lithology=names.get(int(c), f"Cluster {c}"),
                top_depth=scale.depth(y0) if scale else float("nan"),
                bottom_depth=scale.depth(y1) if scale else float("nan"),
                confidence=(scale.confidence if scale else 0.3) * match_conf.get(int(c), 0.5),
            )
            intervals.append(iv)
        res.warnings += check_intervals(intervals, lc, cfg.checks)
        res.intervals = intervals


def write_outputs(doc: Document, res: DocumentResult, out_dir: Path, cfg: Config, *, image: bool = True) -> None:
    """CSV (submission format), diagnostics CSV, and the annotated image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(res.intervals, out_dir / f"{doc.doc_id}.csv")
    write_csv(res.intervals, out_dir / "diagnostics" / f"{doc.doc_id}.csv", with_diagnostics=True)
    if image and res.boxes is not None:
        img = open_image(doc.image_path)
        title = f"{doc.doc_id}  |  {len(res.intervals)} intervals"
        if res.scale:
            title += f"  |  {res.scale.px_per_ft:.2f} px/ft ({res.scale.method}, {res.scale.n_points} labels)"
        view = render_annotated(
            img,
            res.boxes,
            res.intervals,
            max_height_px=cfg.render.max_height_px,
            coverage=cfg.render.coverage,
            title=title,
        )
        view.save(out_dir / f"{doc.doc_id}.png", optimize=True)
