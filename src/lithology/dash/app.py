"""The dashboard's HTTP API (FastAPI). ``lith dash`` serves it together with the built front end."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import duckdb
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lithology.dash import insights as ins
from lithology.dash.ai import AIError, Provider, default_provider, make_provider, provider_infos
from lithology.dash.layout import LayoutView, latest_run
from lithology.dash.paths import Paths
from lithology.dash.store import ingest, live_source, read_summary

log = logging.getLogger(__name__)


def _clean(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    return v


class State:
    def __init__(self, paths: Paths):
        from lithology.config import load_config

        self.paths = paths
        self.unresolved_label = load_config().naming.unresolved_label
        self.con = duckdb.connect(str(paths.db))
        self.lock = threading.RLock()
        self.ask_lock = threading.Lock()
        self.mem = ins.snapshot(self.con)
        self.notes = ins.NoteCache(paths.ai_cache / "notes.json")
        self.answers = ins.AnswerCache(paths.ai_cache / "answers.json")
        self.provider_name = default_provider()
        self._provider: Provider | None = None
        self.jobs = None

    def q(self, sql: str, params: list | None = None) -> list[dict[str, Any]]:
        with self.lock:
            cur = self.con.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        return [{c: _clean(v) for c, v in zip(cols, r, strict=True)} for r in rows]

    @property
    def provider(self) -> Provider:
        if self._provider is None or self._provider.name != self.provider_name:
            try:
                self._provider = make_provider(self.provider_name)
            except AIError:
                self.provider_name = "none"
                self._provider = make_provider("none")
        return self._provider

    def ingest_live(self, doc_id: str, row: dict[str, str]) -> str:
        reg = json.loads((self.paths.out_live / "sources.json").read_text())[doc_id]
        src = live_source(self.paths, doc_id, reg, latest_run(self.paths))
        with self.lock:
            lid = ingest(
                self.con,
                src,
                row or read_summary(self.paths.out_live).get(doc_id),
                self.unresolved_label,
                self.paths.tiles,
            )
            mem = ins.snapshot(self.con)
        with self.ask_lock:
            self.mem = mem
        return lid


class AskBody(BaseModel):
    question: str


class ProviderBody(BaseModel):
    provider: str


class NotesBody(BaseModel):
    correlation: list[str] | None = None


def _log_or_404(st: State, log_id: str) -> dict[str, Any]:
    rows = st.q("SELECT * FROM logs WHERE log_id = ?", [log_id])
    if not rows:
        raise HTTPException(404, "no such log")
    return rows[0]


def create_app(paths: Paths | None = None) -> FastAPI:
    paths = paths or Paths()
    try:
        from dotenv import load_dotenv

        load_dotenv(paths.repo / ".env")
    except ImportError:
        pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from lithology.dash.jobs import Jobs

        st: State = app.state.st
        st.jobs = Jobs(paths, st.ingest_live)
        yield
        st.jobs.shutdown()
        app.state.layout.shutdown()
        st.con.close()

    app = FastAPI(title="Data Underneath", lifespan=lifespan)
    app.state.st = State(paths)
    build_meta = json.loads(paths.build_meta.read_text()) if paths.build_meta.exists() else {}
    app.state.layout = LayoutView(paths, include_holdout=bool(build_meta.get("include_holdout")))

    def st() -> State:
        return app.state.st

    @app.exception_handler(AIError)
    async def _ai_error(_req: Request, exc: AIError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    # ---------------------------------------------------------------- logs

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        build = json.loads(paths.build_meta.read_text()) if paths.build_meta.exists() else {}
        n = st().q("SELECT count(*) AS n FROM logs")[0]["n"]
        return {**build, "logs": n}

    @app.get("/api/logs")
    def logs() -> list[dict[str, Any]]:
        return st().q(
            """SELECT log_id, doc_id, title, well_group, source, kind, depth_ref, scale, top_ft, bottom_ft,
                      n_intervals, n_unresolved FROM logs ORDER BY source DESC, well_group, title, doc_id"""
        )

    @app.get("/api/logs/{log_id}")
    def log_detail(log_id: str) -> dict[str, Any]:
        row = _log_or_404(st(), log_id)
        for k in ("boxes", "boxes_annotated", "boxes_detected", "tiles", "timings"):
            row[k] = json.loads(row[k]) if row[k] else None
        row["warnings"] = [w for w in (row["warnings"] or "").split(" | ") if w]
        row.pop("image_path", None)
        row["rocks"] = st().q(
            """SELECT rock, any_value(color) AS color, any_value(rock_family) AS family, count(*) AS beds,
                      sum(thickness_ft) AS feet, list(DISTINCT cluster_id ORDER BY cluster_id) AS clusters,
                      bool_or(unresolved) AS unresolved, bool_or(lookalike) AS lookalike, min(idx) AS first
               FROM intervals WHERE log_id = ? GROUP BY rock ORDER BY feet DESC NULLS LAST""",
            [log_id],
        )
        legend = st().q("SELECT * FROM legend WHERE log_id = ? ORDER BY used DESC, beds DESC, idx", [log_id])
        tiles_dir = paths.tiles / log_id
        for e in legend:
            sw = tiles_dir / f"sw{e['idx']:03d}.png"
            e["swatch_url"] = f"/tiles/{log_id}/sw{e['idx']:03d}.png" if sw.exists() else None
        row["legend"] = legend
        row["naming"] = st().q("SELECT * FROM naming WHERE log_id = ? ORDER BY grp", [log_id])
        return row

    @app.get("/api/logs/{log_id}/intervals")
    def intervals(log_id: str) -> list[dict[str, Any]]:
        _log_or_404(st(), log_id)
        return st().q(
            """SELECT idx, top_depth, bottom_depth, thickness_ft, top_y, bottom_y, cluster_id, lithology, rock,
                      rock_family, unresolved, lookalike, alternatives, confidence, notes, color
               FROM intervals WHERE log_id = ? ORDER BY idx""",
            [log_id],
        )

    @app.get("/api/logs/{log_id}/csv")
    def csv_file(log_id: str) -> FileResponse:
        row = _log_or_404(st(), log_id)
        out = paths.out_live if row["source"] == "live" else paths.out_batch
        f = out / f"{row['doc_id']}.csv"
        if not f.exists():
            raise HTTPException(404, "CSV not found")
        return FileResponse(f, media_type="text/csv", filename=f.name)

    # ---------------------------------------------------------------- insights + AI

    def _ai_status() -> dict[str, Any]:
        return {
            "active": st().provider_name,
            "model": st().provider.model,
            "providers": [p.__dict__ for p in provider_infos()],
        }

    @app.get("/api/ai")
    def ai_status() -> dict[str, Any]:
        return _ai_status()

    @app.post("/api/ai")
    def ai_switch(body: ProviderBody) -> dict[str, Any]:
        infos = {p.name: p for p in provider_infos()}
        p = infos.get(body.provider)
        if p is None or not p.available:
            raise HTTPException(400, f"provider not available: {p.reason if p else body.provider}")
        st().provider_name = body.provider
        return _ai_status()

    def _cards(corr_ids: list[str] | None, regenerate: bool) -> dict[str, Any]:
        with st().lock:
            con = st().con
            head = ins.headline(con)
            mix = ins.rock_mix(con)
            corr = ins.correlation(con, corr_ids)
        notes = {
            "mix": ins.note("mix", mix, st().provider, st().notes, regenerate=regenerate),
            "correlation": ins.note(
                "correlation", {"logs": corr["logs"]}, st().provider, st().notes, regenerate=regenerate
            ),
        }
        return {
            "headline": {k: _clean(v) for k, v in head.items()},
            "mix": mix,
            "correlation": corr,
            "families": ins.families(),
            "notes": notes,
            "suggested": ins.SUGGESTED,
        }

    @app.get("/api/insights")
    def insights(correlation: str | None = None) -> dict[str, Any]:
        return _cards(correlation.split(",") if correlation else None, regenerate=False)

    @app.post("/api/insights/notes")
    def insights_notes(body: NotesBody) -> dict[str, Any]:
        if st().provider_name == "none":
            raise HTTPException(400, "pick an AI provider first")
        return _cards(body.correlation, regenerate=True)

    @app.post("/api/ask")
    def ask(body: AskBody) -> dict[str, Any]:
        try:
            return ins.ask(body.question, st().provider, st().mem, st().ask_lock, st().answers)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # ---------------------------------------------------------------- live runs

    @app.post("/api/runs")
    async def start_run(files: list[UploadFile], boxes: str = Form("auto")) -> dict[str, Any]:
        """One scan plus an optional box file (.json).

        ``boxes``: "auto" uses the box file when uploaded, else the layout detector; "detect" always detects.
        """
        data = [(f.filename or "upload", await f.read()) for f in files]
        try:
            job_id = st().jobs.submit(data, boxes)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"id": job_id}

    @app.get("/api/runs/{job_id}")
    def run_status(job_id: str) -> dict[str, Any]:
        s = st().jobs.status(job_id)
        if s is None:
            raise HTTPException(404, "no such run")
        return s

    # ---------------------------------------------------------------- layout detection

    def lay() -> LayoutView:
        return app.state.layout

    @app.get("/api/layout")
    def layout_summary() -> dict[str, Any]:
        return lay().summary()

    @app.get("/api/layout/pages/{doc_id}")
    def layout_page(doc_id: str) -> dict[str, Any]:
        d = lay().detail(doc_id)
        if d is None:
            raise HTTPException(404, "no such page (or its detection is still running)")
        return d

    @app.get("/api/layout/pages/{doc_id}/crop")
    def layout_crop(doc_id: str, y0: int, y1: int, w: int = 1400) -> Response:
        png = lay().crop_png(doc_id, y0, y1, w)
        if png is None:
            raise HTTPException(404, "no such page")
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/layout/detect")
    async def layout_detect(file: UploadFile) -> dict[str, Any]:
        try:
            job_id = lay().submit(file.filename or "upload", await file.read())
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"id": job_id}

    @app.get("/api/layout/detect/{job_id}")
    def layout_job(job_id: str) -> dict[str, Any]:
        j = lay().job(job_id)
        if j is None:
            raise HTTPException(404, "no such detection")
        return j

    # ---------------------------------------------------------------- static

    paths.tiles.mkdir(parents=True, exist_ok=True)
    app.mount("/tiles", StaticFiles(directory=paths.tiles), name="tiles")
    if (paths.dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=paths.dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path == "api" or full_path.startswith("api/"):  # never answer a data request with the page
                raise HTTPException(
                    404, f"/{full_path} does not exist on this server: restart the dashboard if the page is newer"
                )
            f = paths.dist / full_path
            if full_path and f.is_file() and paths.dist in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(paths.dist / "index.html")

    return app
