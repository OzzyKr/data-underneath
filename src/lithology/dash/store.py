"""Build the dashboard database: every log's intervals, legend and run stats in one DuckDB file.

Sources are the pipeline's own outputs (``out/`` for the batch run, ``out_live/`` for scans run from the dashboard),
the organisers' bounding boxes, and the scan tiles. Holdout logs (``labels/selection.yaml``) are left out unless
``include_holdout`` is set, so development screenshots never show them.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
from PIL import Image

from lithology.dash.paths import Paths
from lithology.dash.rocks import CHROMA_MIN, SIMILAR, UNRESOLVED, display_name, log_colours, rock_family
from lithology.dash.titles import api_number, scale_label, short_title
from lithology.io import depth_reference, family_of, scale_inches_per_100ft
from lithology.models import Annotation, DocumentBoxes

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE logs (
  log_id VARCHAR PRIMARY KEY, doc_id VARCHAR, source VARCHAR, title VARCHAR, well_group VARCHAR, family VARCHAR,
  api VARCHAR, kind VARCHAR, depth_ref VARCHAR, scale VARCHAR, top_ft DOUBLE, bottom_ft DOUBLE, feet DOUBLE,
  n_intervals INTEGER, n_rocks INTEGER, n_unresolved INTEGER, mean_confidence DOUBLE, px_per_ft DOUBLE,
  scale_points INTEGER, scale_confidence DOUBLE, seconds DOUBLE, peak_rss_mb DOUBLE, timings VARCHAR,
  warnings VARCHAR, image_w INTEGER, image_h INTEGER, boxes VARCHAR, tiles VARCHAR, image_path VARCHAR,
  added VARCHAR, boxes_source VARCHAR, boxes_annotated VARCHAR, boxes_detected VARCHAR
);
CREATE TABLE intervals (
  log_id VARCHAR, idx INTEGER, top_depth DOUBLE, bottom_depth DOUBLE, thickness_ft DOUBLE, top_y INTEGER,
  bottom_y INTEGER, cluster_id INTEGER, lithology VARCHAR, rock VARCHAR, rock_family VARCHAR, unresolved BOOLEAN,
  lookalike BOOLEAN, alternatives VARCHAR, confidence DOUBLE, notes VARCHAR, color VARCHAR
);
CREATE TABLE legend (
  log_id VARCHAR, idx INTEGER, label VARCHAR, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER, is_rock BOOLEAN,
  ocr_confidence DOUBLE, rgb VARCHAR, chroma DOUBLE, used BOOLEAN, beds INTEGER, cluster_id INTEGER, color VARCHAR
);
CREATE TABLE naming (
  log_id VARCHAR, grp INTEGER, decision VARCHAR, reason VARCHAR, confidence DOUBLE, classical VARCHAR,
  template VARCHAR, pretrained VARCHAR, candidates VARCHAR
);
"""
TABLES = ("logs", "intervals", "legend", "naming")


def slug(source: str, doc_id: str) -> str:
    """Stable, URL-safe id (file names hold spaces, '#' and '.PDF')."""
    return hashlib.sha1(f"{source}:{doc_id}".encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Source:
    doc_id: str
    source: str  # "batch" | "live"
    out_dir: Path
    image_path: Path | None
    ann_path: Path | None  # the boxes the run used (batch: the organisers' file)
    uploaded_ann: Path | None = None  # live runs: the box file uploaded with the scan, if any
    layout_run: Path | None = None  # newest `lith layout` run: the detector's boxes for dataset scans

    @property
    def log_id(self) -> str:
        return slug(self.source, self.doc_id)


def holdout_prefixes(paths: Paths) -> list[str]:
    if not paths.selection.exists():
        return []
    data = yaml.safe_load(paths.selection.read_text(encoding="utf-8")) or {}
    return [d["prefix"] for d in data.get("documents", []) if d.get("split") == "holdout"]


def _find_image(images: Path, doc_id: str) -> Path | None:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".jpg", ".jpeg"):
        p = images / f"{doc_id}{ext}"
        if p.exists():
            return p
    return None


def live_source(paths: Paths, doc_id: str, rec: dict[str, Any], layout_run: Path | None = None) -> Source:
    """A live run from its ``out_live/sources.json`` record."""
    # older records have no boxes_source; their "ann" is the uploaded (or dataset) box file
    uploaded = rec.get("uploaded_ann") if "boxes_source" in rec else rec.get("ann")
    return Source(
        doc_id,
        "live",
        paths.out_live,
        Path(rec["image"]),
        Path(rec["ann"]) if rec.get("ann") else None,
        uploaded_ann=Path(uploaded) if uploaded else None,
        layout_run=layout_run,
    )


def discover_sources(paths: Paths, include_holdout: bool = False) -> list[Source]:
    from lithology.dash.layout import latest_run

    hold = [] if include_holdout else holdout_prefixes(paths)
    run = latest_run(paths)
    out: list[Source] = []
    for p in sorted(paths.out_batch.glob("*.csv")):
        if p.name == "run_summary.csv" or any(p.stem.startswith(h) for h in hold):
            continue
        ann = paths.annotations / f"{p.stem}.json"
        image = _find_image(paths.images, p.stem)
        out.append(Source(p.stem, "batch", paths.out_batch, image, ann if ann.exists() else None, layout_run=run))
    reg = paths.out_live / "sources.json"
    if reg.exists():
        for doc_id, rec in json.loads(reg.read_text(encoding="utf-8")).items():
            if (paths.out_live / f"{doc_id}.csv").exists():
                out.append(live_source(paths, doc_id, rec, run))
    return out


def read_summary(out_dir: Path) -> dict[str, dict[str, str]]:
    p = out_dir / "run_summary.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> None


def _norm(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def _hex(rgb: list[int] | None) -> str | None:
    return "#{:02X}{:02X}{:02X}".format(*rgb) if rgb else None


def load_doc(src: Source, summary: dict[str, str] | None, unresolved_label: str, tiles_root: Path) -> dict[str, Any]:
    """Everything the dashboard knows about one log, as one row for ``logs`` and frames for the other tables."""
    diag = src.out_dir / "diagnostics" / f"{src.doc_id}.csv"
    df = pd.read_csv(
        diag if diag.exists() else src.out_dir / f"{src.doc_id}.csv",
        keep_default_na=False,
        na_values=[""],
        dtype={"lithology": str, "notes": str, "status": str, "alternatives": str},
    )
    df["lithology"] = df["lithology"].astype(str)
    notes = df["notes"].fillna("").astype(str) if "notes" in df else pd.Series([""] * len(df))
    legend = []
    lp = src.out_dir / "diagnostics" / f"{src.doc_id}_legend.json"
    if lp.exists():
        legend = json.loads(lp.read_text(encoding="utf-8"))
    naming = []
    np_ = src.out_dir / "diagnostics" / f"{src.doc_id}_naming.json"
    if np_.exists():
        naming = json.loads(np_.read_text(encoding="utf-8"))

    by_label = {_norm(e["label"]): e for e in legend}
    rocks_in_order = []
    for name in dict.fromkeys(df["lithology"]):
        e = by_label.get(_norm(name))
        rocks_in_order.append(
            (display_name(name), tuple(e["rgb"]) if e and e.get("rgb") else None, e["chroma"] if e else 0.0)
        )
    colours = log_colours(rocks_in_order)

    top = pd.to_numeric(df["top_depth"], errors="coerce")
    bottom = pd.to_numeric(df["bottom_depth"], errors="coerce")
    thick = (bottom - top).where(lambda s: s > 0)
    rock = df["lithology"].map(display_name)
    # status column: named | similar | unresolved; older outputs only have the notes
    if "status" in df:
        status = df["status"].fillna("named").astype(str)
        similar = status == "similar"
        unresolved = status == "unresolved"
    else:
        similar = pd.Series([False] * len(df))
        unresolved = (df["lithology"].str.lower() == unresolved_label.lower()) | notes.str.contains(
            "could not differentiate"
        )
    alternatives = df["alternatives"].fillna("").astype(str) if "alternatives" in df else pd.Series([""] * len(df))
    lid = src.log_id
    intervals = pd.DataFrame(
        {
            "log_id": lid,
            "idx": range(len(df)),
            "top_depth": top,
            "bottom_depth": bottom,
            "thickness_ft": thick,
            "top_y": df["top_y"].astype(int),
            "bottom_y": df["bottom_y"].astype(int),
            "cluster_id": pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64"),
            "lithology": df["lithology"],
            "rock": rock,
            "rock_family": [
                UNRESOLVED if u else SIMILAR if m else rock_family(n)
                for n, u, m in zip(df["lithology"], unresolved, similar, strict=True)
            ],
            "unresolved": unresolved,
            "lookalike": similar,
            "alternatives": alternatives,
            "confidence": pd.to_numeric(df["confidence"], errors="coerce") if "confidence" in df else None,
            "notes": notes,
            "color": rock.map(colours),
        }
    )

    used = {_norm(n): n for n in df["lithology"]}
    leg_rows = []
    for i, e in enumerate(legend):
        name = used.get(_norm(e["label"]))
        beds = intervals[intervals["lithology"] == name] if name else intervals.iloc[0:0]
        rgb = e.get("rgb")
        colour = colours.get(display_name(name)) if name else None
        leg_rows.append(
            {
                "log_id": lid,
                "idx": i,
                "label": e["label"],
                "x0": e["swatch"][0],
                "y0": e["swatch"][1],
                "x1": e["swatch"][2],
                "y1": e["swatch"][3],
                "is_rock": bool(e.get("is_rock", True)),
                "ocr_confidence": _num(e.get("ocr_confidence")),
                "rgb": _hex(rgb),
                "chroma": _num(e.get("chroma")) or 0.0,
                "used": name is not None,
                "beds": len(beds),
                "cluster_id": int(beds["cluster_id"].dropna().iloc[0])
                if len(beds) and beds["cluster_id"].notna().any()
                else None,
                "color": colour,
            }
        )
    naming_rows = [
        {
            "log_id": lid,
            "grp": n.get("group"),
            "decision": n.get("decision"),
            "reason": n.get("reason"),
            "confidence": _num(n.get("confidence")),
            "classical": json.dumps(n.get("classical")),
            "template": json.dumps(n.get("template")) if n.get("template") else None,
            "pretrained": json.dumps(n.get("pretrained")) if n.get("pretrained") else None,
            "candidates": json.dumps(n.get("candidates")),
        }
        for n in naming
    ]

    boxes, size = used_boxes(src)
    tiles_meta = tiles_root / lid / "meta.json"
    s = summary or {}
    rock_rows = intervals[~intervals["unresolved"] & ~intervals["lookalike"]]
    is_colour = any(e.get("chroma", 0) >= CHROMA_MIN for e in legend if e.get("is_rock", True))
    row = {
        "log_id": lid,
        "doc_id": src.doc_id,
        "source": src.source,
        "title": short_title(src.doc_id),
        "well_group": None,  # filled once all logs are known
        "family": family_of(src.doc_id),
        "api": api_number(src.doc_id),
        "kind": "colour" if is_colour else ("black and white" if legend else "unknown"),
        "depth_ref": depth_reference(src.doc_id),
        "scale": scale_label(scale_inches_per_100ft(src.doc_id)),
        "top_ft": _num(top.min()),
        "bottom_ft": _num(bottom.max()),
        "feet": _num(thick.sum()) or 0.0,
        "n_intervals": len(intervals),
        "n_rocks": int(rock_rows["rock"].nunique()),
        "n_unresolved": int(intervals["unresolved"].sum()),
        "mean_confidence": _num(intervals["confidence"].mean()) if "confidence" in df else None,
        "px_per_ft": _num(s.get("px_per_ft")),
        "scale_points": int(_num(s.get("scale_points")) or 0),
        "scale_confidence": _num(s.get("scale_conf")),
        "seconds": _num(s.get("seconds")),
        "peak_rss_mb": _num(s.get("peak_rss_mb")),
        "timings": s.get("timings") or None,
        "warnings": s.get("warnings") or None,
        "image_w": size[0] if size else None,
        "image_h": size[1] if size else None,
        "boxes": json.dumps(boxes) if boxes else None,
        "tiles": tiles_meta.read_text(encoding="utf-8") if tiles_meta.exists() else None,
        "image_path": str(src.image_path) if src.image_path else None,
        "added": dt.datetime.now().isoformat(timespec="seconds"),
        "boxes_source": _boxes_source(src),
        "boxes_annotated": _json_or_none(annotated_boxes(src)),
        "boxes_detected": _json_or_none(detected_boxes(src)),
    }
    return {"log": row, "intervals": intervals, "legend": pd.DataFrame(leg_rows), "naming": pd.DataFrame(naming_rows)}


def _json_or_none(v: Any) -> str | None:
    return json.dumps(v) if v else None


def _layout_diag(src: Source) -> dict[str, Any]:
    """The run's layout step record (diagnostics/<doc>_layout.json); {} for runs from before the layout step."""
    try:
        return json.loads((src.out_dir / "diagnostics" / f"{src.doc_id}_layout.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def used_boxes(src: Source) -> tuple[dict | None, tuple[int, int] | None]:
    """The boxes the run used and the image size: detected boxes from diagnostics, else the box file's."""
    lay = _layout_diag(src)
    if lay.get("source") == "detected" and lay.get("boxes") and src.image_path and src.image_path.exists():
        with Image.open(src.image_path) as im:
            return lay["boxes"], im.size
    if src.ann_path and src.ann_path.exists():
        ann = Annotation.from_json(src.ann_path)
        return ann.boxes.to_dict(), (ann.width, ann.height)
    return None, None


def _boxes_source(src: Source) -> str:
    """Where the run's boxes came from: "detector" or "json" (box file, including runs before the layout step)."""
    return "detector" if _layout_diag(src).get("source") == "detected" else "json"


def annotated_boxes(src: Source) -> dict | None:
    """The box file's boxes (batch: the organisers' annotation; live: the file uploaded with the scan), if any."""
    p = src.ann_path if src.source == "batch" else src.uploaded_ann
    try:
        return Annotation.from_json(p).boxes.to_dict() if p and p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def detected_boxes(src: Source) -> dict | None:
    """The detector's boxes: from the run when it detected them, else from the newest ``lith layout`` run."""
    lay = _layout_diag(src)
    if lay.get("source") == "detected" and lay.get("boxes"):
        return lay["boxes"]
    if src.layout_run is None:
        return None
    try:
        return json.loads((src.layout_run / f"{src.doc_id}_layout.json").read_text(encoding="utf-8"))["boxes"]
    except (OSError, ValueError, KeyError):
        return None


def _insert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    con.register("_ins", df.reindex(columns=cols))
    try:
        con.execute(f"INSERT INTO {table} SELECT * FROM _ins")
    finally:
        con.unregister("_ins")


def ingest(
    con: duckdb.DuckDBPyConnection, src: Source, summary: dict[str, str] | None, unresolved_label: str, tiles_root: Path
) -> str:
    """Insert (or replace) one log. Returns its log_id."""
    data = load_doc(src, summary, unresolved_label, tiles_root)
    lid = src.log_id
    for t in TABLES:
        con.execute(f"DELETE FROM {t} WHERE log_id = ?", [lid])
    _insert(con, "logs", pd.DataFrame([data["log"]]))
    for t in ("intervals", "legend", "naming"):
        _insert(con, t, data[t])
    assign_groups(con)
    return lid


def assign_groups(con: duckdb.DuckDBPyConnection) -> None:
    """A readable well label per log: logs of one well (same 10-digit API number) share it."""
    rows = con.execute("SELECT log_id, api, family, title FROM logs").fetchall()
    groups: dict[str, list[tuple[str, str]]] = {}
    for lid, api, fam, title in rows:
        if fam.startswith("series:"):
            key = "series"
        elif api:
            key = "api:" + api[:10]
        else:
            key = "title:" + " ".join(title.lower().split()[:2])
        groups.setdefault(key, []).append((lid, title))
    for key, members in groups.items():
        title = min((t for _, t in members), key=len)
        words = title.split()
        if key == "series":
            label = "D-582 series"
        elif title.startswith("Well ") or len(words) < 2:
            label = title
        elif (
            key.startswith("api:")
            and len(words[0]) >= 4
            and (any(c.isdigit() for c in words[0]) or not words[0].isupper())
        ):
            label = words[0]
        else:
            label = " ".join(words[:2])
        con.executemany("UPDATE logs SET well_group = ? WHERE log_id = ?", [[label, lid] for lid, _ in members])


def _tile_job(args: tuple[str, str, dict, list, str]) -> tuple[str, str | None]:
    lid, image, boxes, legend, out_dir = args
    from lithology.dash.tiles import build_tiles

    try:
        build_tiles(Path(image), DocumentBoxes.from_dict(boxes), legend, Path(out_dir))
        return lid, None
    except Exception as e:  # one bad scan must not stop the build
        return lid, f"{type(e).__name__}: {e}"


def tile_args(src: Source, paths: Paths) -> tuple[str, str, dict, list, str] | None:
    if not (src.image_path and src.image_path.exists()):
        return None
    boxes, _ = used_boxes(src)
    if boxes is None:
        return None
    lp = src.out_dir / "diagnostics" / f"{src.doc_id}_legend.json"
    legend = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    return src.log_id, str(src.image_path), boxes, legend, str(paths.tiles / src.log_id)


def build(
    paths: Paths,
    *,
    include_holdout: bool = False,
    tiles: bool = True,
    rebuild_tiles: bool = False,
    workers: int = 2,
    unresolved_label: str = "Unresolved",
) -> dict[str, Any]:
    paths.cache.mkdir(parents=True, exist_ok=True)
    srcs = discover_sources(paths, include_holdout)
    if tiles:
        jobs = [
            a
            for s in srcs
            if (a := tile_args(s, paths)) and (rebuild_tiles or not (paths.tiles / s.log_id / "meta.json").exists())
        ]
        if jobs:
            log.info("building tiles for %d logs (%d workers)", len(jobs), workers)
            with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
                for lid, err in ex.map(_tile_job, jobs):
                    if err:
                        log.warning("tiles failed for %s: %s", lid, err)
    tmp = paths.db.with_name("logs.build.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(SCHEMA)
    summaries = {paths.out_batch: read_summary(paths.out_batch), paths.out_live: read_summary(paths.out_live)}
    for s in srcs:
        try:
            ingest(con, s, summaries[s.out_dir].get(s.doc_id), unresolved_label, paths.tiles)
        except Exception as e:
            log.warning("skipped %s: %s: %s", s.doc_id, type(e).__name__, e)
    n = con.execute("SELECT count(*) FROM logs").fetchone()[0]
    con.close()
    tmp.replace(paths.db)
    meta = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "include_holdout": include_holdout,
        "logs": n,
        "holdout_hidden": len(holdout_prefixes(paths)) if not include_holdout else 0,
    }
    run_meta = paths.out_batch / "run_meta.json"
    if run_meta.exists():
        meta["pipeline"] = json.loads(run_meta.read_text(encoding="utf-8"))
    paths.build_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
