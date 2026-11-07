"""Run every step for one document, with timing, a memory reading and a guard around each step."""

from __future__ import annotations

import json
import logging
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
from lithology.legend import match_distances, read_legend, swatch_descriptor
from lithology.models import Box, DocumentBoxes, DocumentResult, Interval
from lithology.naming import decide, template_similarity, unresolved_note
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


def _inner(swatch_rgb: np.ndarray) -> np.ndarray:
    """A swatch without its border (central 70%)."""
    h, w = swatch_rgb.shape[:2]
    return swatch_rgb[int(h * 0.15) : int(h * 0.85) or h, int(w * 0.15) : int(w * 0.85) or w]


def _cluster_patches(column: np.ndarray, filled: list, ids: np.ndarray, c: int, block: int, n: int = 4) -> list:
    """Pixel crops of a group's thickest intervals, trimmed away from their boundaries."""
    members = sorted((s for s, i in zip(filled, ids, strict=True) if i == c), key=lambda s: s.top - s.bottom)[:n]
    out = []
    for s in members:
        a, b = (s.top + block, s.bottom - block) if s.bottom - s.top > 3 * block else (s.top, s.bottom)
        out.append(column[a:b])
    return out


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
        notes: dict[int, str] = {}
        rocks = [e for e in entries if e.is_rock] or entries  # never name a bed after a drilling symbol
        if rocks and len(ids):
            entries = rocks
            sw_rgb = [crop_rgb(img, e.swatch) for e in entries]
            sw = np.array([swatch_descriptor(s, cfg.segment.ink_luma) for s in sw_rgb])
            n_clusters = int(ids.max()) + 1
            cdesc = np.array(
                [_cluster_match_descriptor(column, filled, ids, c, seg.block_px, cfg) for c in range(n_clusters)]
            )
            dist = match_distances(cdesc, sw)
            patches = [_cluster_patches(column, filled, ids, c, seg.block_px) for c in range(n_clusters)]
            inner = [_inner(s) for s in sw_rgb]
            tmpl = template_similarity(patches, inner)
            decisions = [decide(dist[c], None, cfg.naming, tmpl[c]) for c in range(n_clusters)]
            # Ask the pretrained model only where it can change the answer: groups the pixel template and the
            # classical matcher left open. Settled groups skip it (it is the slowest step).
            open_groups = [c for c, d in enumerate(decisions) if d.reason == "classical match too close to call"]
            sim = np.full((n_clusters, len(entries)), np.nan)
            if cfg.ai.tier >= 2 and open_groups:
                try:
                    from lithology.ai.embed import similarity

                    sim[open_groups] = similarity([patches[c] for c in open_groups], inner)
                    for c in open_groups:
                        decisions[c] = decide(dist[c], sim[c], cfg.naming, tmpl[c])
                except Exception as e:  # the second opinion is optional: degrade to classical only
                    res.warnings.append(f"pretrained second opinion unavailable ({type(e).__name__}: {e})")
            for c, d in enumerate(decisions):
                a = np.argsort(dist[c])[:2]
                rec = {
                    "group": c,
                    "classical": [[entries[i].label, round(float(dist[c][i]), 3)] for i in a],
                    "decision": entries[d.swatch].label if d.swatch is not None else cfg.naming.unresolved_label,
                    "candidates": [entries[i].label for i in d.candidates],
                    "reason": d.reason,
                    "confidence": d.confidence,
                }
                if np.isfinite(tmpl[c]).all():
                    best2 = np.argsort(-tmpl[c])[:2]
                    rec["template"] = [[entries[i].label, round(float(tmpl[c][i]), 3)] for i in best2]
                if np.isfinite(sim[c]).all():
                    b = np.argsort(-sim[c])[:2]
                    rec["pretrained"] = [[entries[i].label, round(float(sim[c][i]), 3)] for i in b]
                res.naming.append(rec)
            # Groups the legend calls the same rock become one group (ids agree with names, touching beds merge).
            # Unresolved groups with the same candidates share one id: the same ambiguity is the same pattern.
            key_of = {
                c: ("rock", d.swatch) if d.swatch is not None else ("open", tuple(sorted(d.candidates)))
                for c, d in enumerate(decisions)
            }
            new_id: dict[tuple, int] = {}
            for c in ids:
                new_id.setdefault(key_of[int(c)], len(new_id))
            remap = {c: new_id[key_of[c]] for c in range(n_clusters) if key_of[c] in new_id}
            for c, d in enumerate(decisions):
                if c not in remap:
                    continue
                r = remap[c]
                if d.swatch is not None:
                    names[r] = entries[d.swatch].label
                else:
                    # abstain: say so in the CSV; otherwise write the best guess but keep the uncertainty note
                    names[r] = cfg.naming.unresolved_label if cfg.naming.abstain else entries[d.candidates[0]].label
                    notes[r] = unresolved_note([entries[i].label for i in d.candidates])
                match_conf[r] = min(match_conf.get(r, 1.0), d.confidence)
            ids = np.array([remap[int(c)] for c in ids])
            n_open = sum(d.swatch is None for c, d in enumerate(decisions) if c in remap)
            if n_open:
                res.warnings.append(f"{n_open} rock group(s) could not be named with confidence")
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
                notes=[notes[int(c)]] if int(c) in notes else [],
            )
            intervals.append(iv)
        res.warnings += check_intervals(intervals, lc, cfg.checks)
        res.intervals = intervals


def write_outputs(doc: Document, res: DocumentResult, out_dir: Path, cfg: Config, *, image: bool = True) -> None:
    """CSV (submission format), diagnostics CSV, and the annotated image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(res.intervals, out_dir / f"{doc.doc_id}.csv")
    write_csv(res.intervals, out_dir / "diagnostics" / f"{doc.doc_id}.csv", with_diagnostics=True)
    (out_dir / "diagnostics" / f"{doc.doc_id}_naming.json").write_text(json.dumps(res.naming, indent=1))
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
