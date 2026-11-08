"""Run every step for one document, with timing, a memory reading and a guard around each step."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lithology.checks import check_intervals
from lithology.cluster import cluster_intervals, merge_adjacent
from lithology.config import Config
from lithology.features import match_descriptor, stroke_descriptor, stroke_distances
from lithology.io import Document, crop_rgb, image_dpi, open_image
from lithology.legend import match_distances, read_legend, swatch_colour, swatch_descriptor
from lithology.models import Box, DocumentBoxes, DocumentResult, Interval
from lithology.naming import (
    Naming,
    best_guess,
    classify_flagged,
    decide,
    fill_lab,
    identical_swatches,
    similar_note,
    template_similarity,
    unresolved_note,
)
from lithology.render import render_annotated, write_csv
from lithology.scale import estimate_scale, prior_px_per_ft
from lithology.segment import core_box_x, interval_descriptors, segment_column, snap_to_gap

log = logging.getLogger(__name__)

DEFAULT_PX_PER_FT = 10.0  # only used for block sizes when neither OCR nor filename/DPI gives a scale


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux; peak working set on Windows)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]  # fmt: skip

        c = _Counters()
        c.cb = ctypes.sizeof(c)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e6
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


StepHook = Callable[[str], None]


@contextmanager
def _timed(timings: dict[str, float], name: str, on_step: StepHook | None = None) -> Generator[None, None, None]:
    if on_step is not None:
        on_step(name)  # progress for live runs (the dashboard); must never break the pipeline
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


def process(
    doc: Document, cfg: Config, boxes: DocumentBoxes | None = None, on_step: StepHook | None = None
) -> DocumentResult:
    """Extract intervals for one document. Never raises: errors are recorded on the result.

    ``on_step(name)`` is called as each step starts (load, scale, segment, cluster, legend, assemble).
    """
    res = DocumentResult(doc_id=doc.doc_id, intervals=[])

    def hook(name: str) -> None:
        if on_step is None:
            return
        try:
            on_step(name)
        except Exception:  # a progress reporter must not fail the document
            log.debug("step hook failed", exc_info=True)

    try:
        _process(doc, cfg, boxes, res, hook)
    except Exception as e:  # the batch must survive any single document
        res.error = f"{type(e).__name__}: {e}"
        log.error("%s failed: %s\n%s", doc.doc_id, res.error, traceback.format_exc())
    res.peak_rss_mb = round(peak_rss_mb(), 1)
    return res


def _layout_step(doc: Document, img, cfg: Config, boxes: DocumentBoxes | None) -> tuple[DocumentBoxes, dict]:
    """Step 1: find the three boxes.

    Boxes passed in win. With ``layout.source: annotation`` (default) the box file is used when it exists;
    otherwise the detector runs, and a box file, if present, only scores it.
    """
    if boxes is not None:
        return boxes, {"source": "supplied"}
    ann = doc.annotation()
    if cfg.layout.source == "annotation" and ann is not None:
        return ann.boxes, {"source": "annotation"}
    from lithology.layout import detect_boxes

    lay = detect_boxes(img, cfg.layout)
    info: dict = {
        "source": "detected",
        "boxes": lay.boxes.to_dict(),
        "confidence": lay.confidence,
        "low_confidence": [k for k, v in lay.confidence.items() if v < 0.5],
    }
    if ann is not None:
        a, d = ann.boxes.to_dict(), lay.boxes.to_dict()
        info["iou_vs_annotation"] = {k: round(Box.from_seq(d[k]).iou(Box.from_seq(a[k])), 3) for k in a}
    return lay.boxes, info


def _process(
    doc: Document, cfg: Config, boxes: DocumentBoxes | None, res: DocumentResult, hook: StepHook | None = None
) -> None:
    t = res.timings_s
    with _timed(t, "load", hook):
        img = open_image(doc.image_path)

    with _timed(t, "layout", hook):  # step 1: depth column, lithology column and legend
        boxes, res.layout = _layout_step(doc, img, cfg, boxes)
        if res.layout.get("low_confidence"):
            res.warnings.append(f"layout: low confidence for {res.layout['low_confidence']}")
        res.boxes = boxes
        lc = boxes.lithology_column.clip(*img.size)

    with _timed(t, "scale", hook):
        scale, warns = estimate_scale(img, boxes.depth_column, doc.doc_id, cfg.scale)
        res.scale = scale
        res.warnings += warns
        px_per_ft = scale.px_per_ft if scale else (prior_px_per_ft(img, doc.doc_id) or DEFAULT_PX_PER_FT)

    with _timed(t, "segment", hook):
        xa, xb = core_box_x(lc.x0, lc.x1, cfg.segment.core_width_fraction)
        column = crop_rgb(img, Box(xa, lc.y0, xb, lc.y1))
        seg = segment_column(column, px_per_ft, cfg.segment)
        filled = [s for s in seg.segments if not s.blank]

    with _timed(t, "cluster", hook):
        desc = interval_descriptors(seg, filled) if filled else np.zeros((0, 12))
        ids = cluster_intervals(desc, seg.scaler, cfg.cluster)

    with _timed(t, "legend", hook):
        entries = []
        try:
            lg = boxes.legend.clip(*img.size)
            legend_rgb = crop_rgb(img, lg)
            entries = read_legend(legend_rgb, (lg.x0, lg.y0), image_dpi(img) or 200.0, cfg.legend)
            for e in entries:  # fill colour of every swatch, for the dashboard's rock colours
                sw = e.swatch
                e.rgb, e.chroma = swatch_colour(
                    legend_rgb[sw.y0 - lg.y0 : sw.y1 - lg.y0, sw.x0 - lg.x0 : sw.x1 - lg.x0], cfg.segment.ink_luma
                )
        except Exception as e:  # a legend failure must not lose the intervals
            res.warnings.append(f"legend read failed: {type(e).__name__}: {e}")
        res.legend = entries
        names: dict[int, str] = {}
        match_conf: dict[int, float] = {}
        notes: dict[int, str] = {}
        status: dict[int, str] = {}
        alternatives: dict[int, list[str]] = {}
        paper_ids: set[int] = set()  # groups that are paper or a text box over the column, not beds
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
            # ask the pretrained model only for groups the template and classical matcher left open (slowest step)
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
            # colour veto: a confident name must share the bed's fill colour. A white group named after a coloured
            # swatch is paper or a text box (dropped); any other clash becomes a flagged guess.
            sw_fill = np.array([fill_lab(s, cfg.segment.ink_luma) for s in inner])
            paper: set[int] = set()
            if np.mean([e.chroma > 15 for e in entries]) >= 0.3:
                for c, d in enumerate(decisions):
                    parts = [p for p in patches[c] if p.size]
                    if d.swatch is None or not parts:
                        continue
                    g = np.mean([fill_lab(p, cfg.segment.ink_luma) for p in parts], axis=0)
                    if np.linalg.norm(g - sw_fill[d.swatch]) <= cfg.naming.colour_gate:
                        continue
                    if g[0] > 90 and np.hypot(g[1], g[2]) < 8 and entries[d.swatch].chroma > 15:
                        paper.add(c)
                    why = "name contradicts the bed's colour"
                    decisions[c] = Naming(None, cfg.naming.confidence_unresolved, d.candidates, why)
            # Groups no opinion settled still get a flagged best guess (see lithology.naming.best_guess).
            labels = [e.label for e in entries]
            identical = identical_swatches(inner)
            guesses: dict[int, tuple[int, float, str, bool]] = {}
            bed_colour_ok: dict[int, np.ndarray | None] = {}
            strokes = np.full((n_clusters, len(entries)), np.nan)
            open_now = [c for c, d in enumerate(decisions) if d.swatch is None]
            if cfg.naming.use_strokes and open_now:
                sw_strokes = np.array([stroke_descriptor(s, cfg.segment.ink_luma) for s in inner])
                for c in open_now:
                    parts = [p for p in patches[c] if p.shape[0] >= 8 and p.shape[1] >= 8]
                    if parts:
                        cd = np.median([stroke_descriptor(p, cfg.segment.ink_luma) for p in parts], axis=0)
                        strokes[c] = stroke_distances(cd[None], sw_strokes)[0]
            for c in open_now:
                d = decisions[c]
                s_c = sim[c] if np.isfinite(sim[c]).all() else None
                k_c = strokes[c] if np.isfinite(strokes[c]).all() else None
                cands = d.candidates + ([int(np.argmin(k_c))] if k_c is not None else [])
                parts = [p for p in patches[c] if p.size]
                colour_ok = None
                if parts:
                    g_fill = np.mean([fill_lab(p, cfg.segment.ink_luma) for p in parts], axis=0)
                    colour_ok = np.linalg.norm(sw_fill - g_fill, axis=1) <= cfg.naming.colour_gate
                bed_colour_ok[c] = colour_ok
                guesses[c] = best_guess(cands, labels, dist[c], s_c, tmpl[c], identical, cfg.naming, k_c, colour_ok)
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
                if c in guesses:
                    rec["guess"], rec["guess_reason"] = entries[guesses[c][0]].label, guesses[c][2]
                if np.isfinite(strokes[c]).all():
                    k = np.argsort(strokes[c])[:2]
                    rec["strokes"] = [[entries[i].label, round(float(strokes[c][i]), 3)] for i in k]
                if np.isfinite(tmpl[c]).any():
                    best2 = np.argsort(-np.nan_to_num(tmpl[c], nan=-1.0))[:2]
                    rec["template"] = [[entries[i].label, round(float(tmpl[c][i]), 3)] for i in best2]
                if np.isfinite(sim[c]).all():
                    b = np.argsort(-sim[c])[:2]
                    rec["pretrained"] = [[entries[i].label, round(float(sim[c][i]), 3)] for i in b]
                res.naming.append(rec)
            # Groups the legend calls the same rock become one group (ids agree with names, touching beds merge).
            # Unresolved groups with the same candidates share one id: the same ambiguity is the same pattern.
            key_of = {
                c: ("paper", c)
                if c in paper
                else ("rock", d.swatch)
                if d.swatch is not None
                else (("guess" if guesses[c][3] else "rock"), guesses[c][0])
                for c, d in enumerate(decisions)
            }
            new_id: dict[tuple, int] = {}
            for c in ids:
                new_id.setdefault(key_of[int(c)], len(new_id))
            remap = {c: new_id[key_of[c]] for c in range(n_clusters) if key_of[c] in new_id}
            for c, d in enumerate(decisions):
                if c not in remap or c in paper:
                    continue
                r = remap[c]
                if d.swatch is not None:
                    names[r] = entries[d.swatch].label
                else:
                    # abstain: say so in the CSV; otherwise write the best guess but keep the uncertainty note
                    pick, conf, _, flagged = guesses[c]
                    if flagged:
                        names[r] = cfg.naming.unresolved_label if cfg.naming.abstain else entries[pick].label
                        ok = bed_colour_ok.get(c)
                        st, alts = classify_flagged(
                            pick, d.candidates, sw_fill, ok is None or bool(ok[pick]), cfg.naming.colour_gate
                        )
                        status[r], alternatives[r] = st, [entries[i].label for i in alts]
                        notes[r] = (similar_note if st == "similar" else unresolved_note)(alternatives[r])
                    else:
                        names[r] = entries[pick].label
                    match_conf[r] = min(match_conf.get(r, 1.0), conf)
                    continue
                match_conf[r] = min(match_conf.get(r, 1.0), d.confidence)
            ids = np.array([remap[int(c)] for c in ids])
            paper_ids = {remap[c] for c in paper if c in remap}
            if paper_ids:
                res.warnings.append(f"{len(paper)} group(s) dropped: paper or a text box over the column")
            n_open = sum(d.swatch is None and guesses[c][3] for c, d in enumerate(decisions) if c in remap)
            if n_open:
                res.warnings.append(f"{n_open} rock group(s) could not be named with confidence")
        else:
            res.warnings.append("no legend entries read; intervals named by cluster")
        del column

    with _timed(t, "assemble", hook):
        merged = merge_adjacent([s.top for s in filled], [s.bottom for s in filled], ids.tolist())
        merged = [m for m in merged if m[2] not in paper_ids]
        if cfg.segment.snap_to_gap:  # final bed edges only: grouping above is unchanged
            merged = [list(m) for m in merged]
            for a, b in zip(merged[:-1], merged[1:], strict=True):
                lo, hi = a[0] + 2, b[1] - 2
                if a[1] == b[0] and hi > lo:  # touching beds of different groups, not slivers
                    edge = snap_to_gap(seg.stats, seg.scaler, a[1], seg.block_px, seg.block_px)
                    a[1] = b[0] = min(max(edge, lo), hi)
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
                status=status.get(int(c), "named"),
                alternatives=alternatives.get(int(c), []),
            )
            intervals.append(iv)
        res.warnings += check_intervals(intervals, lc, cfg.checks)
        res.intervals = intervals


def write_outputs(doc: Document, res: DocumentResult, out_dir: Path, cfg: Config, *, image: bool = True) -> None:
    """CSV (submission format), diagnostics CSV, and the annotated image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(res.intervals, out_dir / f"{doc.doc_id}.csv")
    write_csv(res.intervals, out_dir / "diagnostics" / f"{doc.doc_id}.csv", with_diagnostics=True)
    (out_dir / "diagnostics" / f"{doc.doc_id}_naming.json").write_text(
        json.dumps(res.naming, indent=1), encoding="utf-8"
    )
    legend = [
        {
            "label": e.label,
            "swatch": list(e.swatch.as_tuple()),
            "is_rock": e.is_rock,
            "ocr_confidence": round(e.ocr_confidence, 3),
            "rgb": list(e.rgb) if e.rgb else None,
            "chroma": round(e.chroma, 1),
        }
        for e in res.legend
    ]
    (out_dir / "diagnostics" / f"{doc.doc_id}_legend.json").write_text(json.dumps(legend, indent=1), encoding="utf-8")
    if res.layout:
        (out_dir / "diagnostics" / f"{doc.doc_id}_layout.json").write_text(
            json.dumps(res.layout, indent=1), encoding="utf-8"
        )
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
