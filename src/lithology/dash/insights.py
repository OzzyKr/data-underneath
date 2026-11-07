"""Insights across all logs: fixed-SQL cards (always correct numbers), AI notes on them, and "Ask the data".

The AI never computes a number shown on a card. For questions it writes read-only SQL against an isolated
in-memory copy of the tables (no file access, no writes), and every answer shows the SQL it ran.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import duckdb

from lithology.dash.ai import AIError, Provider
from lithology.dash.rocks import FAMILY_COLOURS, FAMILY_ORDER

SUGGESTED = [
    "Which well has the most sand below 10,000 ft?",
    "Where is the salt thickest?",
    "Which logs did the pipeline find hardest to name?",
]

# ------------------------------------------------------------------------------------------------ cards


def _rows(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def headline(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    h = _rows(
        con,
        """SELECT count(*) AS logs, sum(n_intervals) AS beds, sum(feet) AS feet,
                  count(DISTINCT well_group) AS wells, avg(seconds) AS seconds_per_log,
                  max(peak_rss_mb) AS peak_mb FROM logs""",
    )[0]
    r = _rows(
        con,
        """SELECT count(DISTINCT lower(rock)) FILTER (WHERE NOT unresolved) AS rock_names,
                  sum(thickness_ft) FILTER (WHERE unresolved) / nullif(sum(thickness_ft), 0) AS unresolved_share
           FROM intervals""",
    )[0]
    return {**h, **r}


def _one_log_per_well(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Wells often have several logs of the same hole (1 in / 5 in, MD / TVD): keep the one covering most feet."""
    return _rows(
        con,
        """SELECT log_id, title, well_group, top_ft, bottom_ft, feet, kind FROM (
             SELECT *, row_number() OVER (PARTITION BY well_group ORDER BY feet DESC) AS k FROM logs WHERE feet > 0
           ) WHERE k = 1 ORDER BY feet DESC""",
    )


def rock_mix(con: duckdb.DuckDBPyConnection, limit: int = 10) -> list[dict[str, Any]]:
    wells = _one_log_per_well(con)[:limit]
    out = []
    for w in wells:
        parts = _rows(
            con,
            """SELECT rock_family AS family, sum(thickness_ft) AS ft FROM intervals
               WHERE log_id = ? AND thickness_ft > 0 GROUP BY 1""",
            [w["log_id"]],
        )
        total = sum(p["ft"] for p in parts) or 1.0
        shares = sorted(
            ({"family": p["family"], "share": p["ft"] / total, "ft": p["ft"]} for p in parts),
            key=lambda s: FAMILY_ORDER.index(s["family"]) if s["family"] in FAMILY_ORDER else 99,
        )
        out.append(
            {"log_id": w["log_id"], "well": w["well_group"], "title": w["title"], "feet": w["feet"], "shares": shares}
        )
    return out


def correlation_candidates(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    return _one_log_per_well(con)


def correlation(con: duckdb.DuckDBPyConnection, log_ids: list[str] | None = None) -> dict[str, Any]:
    cands = correlation_candidates(con)
    ids = [i for i in (log_ids or []) if any(c["log_id"] == i for c in cands)] or [c["log_id"] for c in cands[:4]]
    logs = []
    for lid in ids[:6]:
        meta = next(c for c in cands if c["log_id"] == lid)
        rows = _rows(
            con,
            """SELECT top_depth AS top, bottom_depth AS bottom, rock_family AS family FROM intervals
               WHERE log_id = ? AND thickness_ft > 0 ORDER BY top_depth""",
            [lid],
        )
        runs: list[dict[str, Any]] = []
        for r in rows:  # merge touching beds of one family: fewer shapes, same picture
            if runs and runs[-1]["family"] == r["family"] and abs(runs[-1]["bottom"] - r["top"]) < 1.0:
                runs[-1]["bottom"] = r["bottom"]
            else:
                runs.append(dict(r))
        logs.append({"log_id": lid, "well": meta["well_group"], "title": meta["title"], "runs": runs})
    return {
        "logs": logs,
        "candidates": [{"log_id": c["log_id"], "well": c["well_group"], "title": c["title"]} for c in cands],
    }


def families() -> list[dict[str, str]]:
    return [{"family": f, "color": FAMILY_COLOURS[f]} for f in FAMILY_ORDER]


# ------------------------------------------------------------------------------------------------ notes


NOTE_SYSTEM = (
    "You write one short note under a chart in an engineering dashboard about digitised well logs. "
    "The data was extracted from scanned mud logs by a computer-vision pipeline; 'Unresolved' means it could not "
    "tell two rocks apart. Write one or two plain sentences, at most 45 words, for geoscientists. Use exact numbers "
    "from the data, with units. No hype and nothing the data does not show. If a value looks implausible (for "
    "example one rock making up almost a whole well), say it is worth checking against the scan. Plain text only."
)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _summary(card: str, data: Any) -> str:
    """A computed, AI-free note used when no AI note is cached for this data."""
    if card == "mix" and data:
        top = max(
            ((w["well"], s["family"], s["share"]) for w in data for s in w["shares"] if s["family"] != "Unresolved"),
            key=lambda t: t[2],
            default=None,
        )
        sand = max(data, key=lambda w: next((s["share"] for s in w["shares"] if s["family"] == "Sand"), 0))
        sand_share = next((s["share"] for s in sand["shares"] if s["family"] == "Sand"), 0)
        parts = []
        if top:
            parts.append(f"The largest single share is {top[1].lower()} in {top[0]} ({_pct(top[2])}).")
        if sand_share:
            parts.append(f"The sandiest well is {sand['well']} ({_pct(sand_share)} sand).")
        return " ".join(parts)
    if card == "correlation" and data:
        deepest = max(data["logs"], key=lambda w: max((r["bottom"] for r in w["runs"]), default=0))
        bottom = max((r["bottom"] for r in deepest["runs"]), default=0)
        return f"{len(data['logs'])} wells on one depth axis; the deepest, {deepest['well']}, reaches {bottom:,.0f} ft."
    return ""


def _data_key(card: str, data: Any) -> str:
    return hashlib.sha1((card + json.dumps(data, sort_keys=True, default=str)).encode()).hexdigest()[:12]


def _compact(card: str, data: Any) -> Any:
    """What the model sees for a card: rounded, no ids."""
    if card == "mix":
        return [
            {"well": w["well"], "log": w["title"], "feet": round(w["feet"]),
             "shares": {s["family"]: round(100 * s["share"], 1) for s in w["shares"]}}
            for w in data
        ]  # fmt: skip
    if card == "correlation":
        out = []
        for w in data["logs"]:
            thick = [r for r in w["runs"] if r["bottom"] - r["top"] >= 200]
            out.append(
                {"well": w["well"], "log": w["title"],
                 "units_200ft_plus": [[r["family"], round(r["top"]), round(r["bottom"])] for r in thick][:40]}
            )  # fmt: skip
        return out
    return data


class NoteCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            return self._load().get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self.lock:
            d = self._load()
            d[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d, indent=1))


def note(card: str, data: Any, provider: Provider, cache: NoteCache, *, regenerate: bool = False) -> dict[str, Any]:
    """The note for a card: cached AI note for this exact data, a fresh one if asked, else a computed summary."""
    key = f"{card}:{_data_key(card, data)}"
    cached = cache.get(key)
    if cached and not regenerate:
        return {**cached, "cached": True}
    if regenerate and provider.name != "none":
        text = provider.complete(NOTE_SYSTEM, f"Chart: {card}\nData (JSON):\n{json.dumps(_compact(card, data))}")
        rec = {"text": text, "source": "ai", "provider": provider.name, "model": provider.model,
               "at": dt.datetime.now().isoformat(timespec="seconds")}  # fmt: skip
        cache.put(key, rec)
        return {**rec, "cached": False}
    return {"text": _summary(card, data), "source": "computed", "provider": None, "model": None, "cached": False}


# ------------------------------------------------------------------------------------------------ ask

_STATEMENT = re.compile(r"^\s*(select|with)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|create|drop|alter|attach|detach|copy|export|import|install|load|pragma|set|reset|"
    r"call|checkpoint|vacuum|use|begin|commit|rollback|grant|revoke|truncate|summarize)\b"
    r"|\b(read_\w+|glob|sniff_csv|parquet_\w+|getenv)\s*\(",
    re.I,
)
_FILE_LITERAL = re.compile(r"'[^']*\.(csv|tsv|parquet|json|txt|duckdb|db|xlsx?)'", re.I)
MAX_ROWS = 200
TIMEOUT_S = 10.0


def check_sql(sql: str) -> str:
    """Only one read-only SELECT/WITH statement, no file access. Raises ValueError otherwise."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("only one statement is allowed")
    if not _STATEMENT.match(s):
        raise ValueError("only SELECT (or WITH ... SELECT) queries are allowed")
    if _FORBIDDEN.search(s) or _FILE_LITERAL.search(s):
        raise ValueError("this query uses a statement or function that is not allowed (read-only, no files)")
    return s


AI_LOG_COLUMNS = (
    "log_id, title, well_group, source, kind, depth_ref, scale, top_ft, bottom_ft, feet, n_intervals, n_rocks, "
    "n_unresolved, mean_confidence, px_per_ft, scale_confidence, seconds, peak_rss_mb"
)


def snapshot(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """An isolated, in-memory copy of the tables for AI-written SQL: no file system, no network, no writes back."""
    mem = duckdb.connect(":memory:")
    queries = {
        "logs": f"SELECT {AI_LOG_COLUMNS} FROM logs",
        "intervals": "SELECT log_id, idx, top_depth, bottom_depth, thickness_ft, top_y, bottom_y, cluster_id, "
        "lithology, rock, rock_family, unresolved, confidence, notes FROM intervals",
        "legend": "SELECT log_id, label, is_rock, used, beds, cluster_id FROM legend",
    }
    for name, q in queries.items():
        df = con.execute(q).df()
        mem.register("_src", df)
        mem.execute(f"CREATE TABLE {name} AS SELECT * FROM _src")
        mem.unregister("_src")
    mem.execute("SET enable_external_access = false")
    mem.execute("SET lock_configuration = true")
    return mem


def run_sql(mem: duckdb.DuckDBPyConnection, sql: str, lock: threading.Lock) -> dict[str, Any]:
    s = check_sql(sql)
    with lock:
        timer = threading.Timer(TIMEOUT_S, mem.interrupt)
        timer.start()
        try:
            cur = mem.execute(s)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
        finally:
            timer.cancel()
    truncated = len(rows) > MAX_ROWS
    rows = [[_jsonable(v) for v in r] for r in rows[:MAX_ROWS]]
    return {"columns": cols, "rows": rows, "truncated": truncated}


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return None if v != v else round(v, 4)
    if isinstance(v, int | str | bool) or v is None:
        return v
    return str(v)


ASK_SYSTEM = """You answer questions about digitised well logs for an engineering demo. A computer-vision pipeline read
the lithology column of each scanned mud log (US Gulf of Mexico offshore wells and some onshore wells) and named each
bed from the log's own legend. 'Unresolved' beds are ones the pipeline could not name with confidence.

You can query three DuckDB tables with the run_sql tool:

logs: one row per log.
  log_id (internal id - never show it), title (short log name), well_group (well name; several logs can belong to one
  well, e.g. 1 in and 5 in scales or MD and TVD versions - avoid double counting by picking one log per well when
  comparing wells), source ('batch' or 'live'), kind ('colour' or 'black and white'), depth_ref ('MD' or 'TVD' or
  NULL), scale, top_ft, bottom_ft, feet (logged thickness), n_intervals, n_rocks, n_unresolved, mean_confidence (0-1),
  px_per_ft, scale_confidence (0-1), seconds (pipeline run time), peak_rss_mb (pipeline peak memory).
intervals: one row per bed (interval) in a log.
  log_id, idx (order from the top), top_depth, bottom_depth (feet), thickness_ft (NULL if depth unknown), top_y,
  bottom_y (pixel rows in the scan), cluster_id, lithology (name as written in the CSV), rock (display name),
  rock_family (one of Sand, Silt, Shale, Clay, Limestone, Dolomite, Salt, Coal, Volcanic, Conglomerate, Other,
  Unresolved), unresolved (boolean), confidence (0-1), notes.
legend: one row per legend entry read from a log.
  log_id, label, is_rock, used (appears in the log's beds), beds, cluster_id.

How to work:
- Get every number from run_sql. Never invent or estimate numbers.
- Only SELECT queries. Aggregate and keep results small (LIMIT 50 or fewer rows).
- Compare rocks across logs with rock_family; sum thickness_ft for "how much"; count rows for "how many beds".
- Name logs by title and wells by well_group.
- When you have the answer, call show_answer exactly once. Its text is 1-4 plain sentences with exact numbers and
  units. Mention uncertainty when it matters (for example a large unresolved share or low confidence). Add a chart
  only when it helps, built from the columns of your last query.
- If the data cannot answer the question, say so plainly in show_answer."""

ASK_TOOLS = [
    {
        "name": "run_sql",
        "description": "Run one read-only DuckDB SELECT query on the logs, intervals and legend tables. Returns up "
        "to 50 rows as JSON (columns + rows).",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT or WITH ... SELECT statement"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "show_answer",
        "description": "Show the final answer to the user. Call once, at the end.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "1-4 plain sentences with exact numbers and units"},
                "chart": {
                    "type": "string",
                    "enum": ["none", "bar", "line", "table"],
                    "description": "Chart of the last query's rows, or none",
                },
                "x": {"type": "string", "description": "Column for the category or x axis (if chart)"},
                "y": {"type": "array", "items": {"type": "string"}, "description": "Numeric column(s) (if chart)"},
                "title": {"type": "string", "description": "Short chart title (if chart)"},
            },
            "required": ["text", "chart"],
            "additionalProperties": False,
        },
    },
]


class AnswerCache(NoteCache):
    @staticmethod
    def key(question: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def ask(
    question: str, provider: Provider, mem: duckdb.DuckDBPyConnection, lock: threading.Lock, cache: AnswerCache
) -> dict[str, Any]:
    question = question.strip()[:500]
    if not question:
        raise ValueError("empty question")
    key = cache.key(question)
    if provider.name == "none":
        hit = cache.get(key)
        if hit:
            return {**hit, "cached": True}
        raise AIError("AI is offline. Pick a provider in the top bar, or ask a question that was answered before.")
    t0 = time.perf_counter()
    queries: list[str] = []
    last: dict[str, Any] = {}
    final: dict[str, Any] = {}

    def execute(name: str, args: dict[str, Any]) -> tuple[str, bool, bool]:
        if name == "run_sql":
            sql = str(args.get("sql", ""))
            try:
                res = run_sql(mem, sql, lock)
            except (ValueError, duckdb.Error) as e:
                return f"Error: {e}", True, False
            queries.append(check_sql(sql))
            last.clear()
            last.update(res)
            preview = {
                "columns": res["columns"],
                "rows": res["rows"][:50],
                "truncated": res["truncated"] or len(res["rows"]) > 50,
            }
            return json.dumps(preview, default=str), False, False
        if name == "show_answer":
            final.update(args)
            return "shown", False, True
        return f"Error: unknown tool {name}", True, False

    text = provider.tool_loop(ASK_SYSTEM, question, ASK_TOOLS, execute)
    answer = str(final.get("text") or text or "").strip()
    if not answer:
        raise AIError("the model returned no answer")
    chart = None
    kind = final.get("chart", "none")
    if kind in ("bar", "line", "table") and last.get("rows"):
        cols = last["columns"]
        x = final.get("x") if final.get("x") in cols else cols[0]
        ys = [y for y in (final.get("y") or []) if y in cols and y != x]
        if kind == "table" or ys:
            chart = {
                "type": kind,
                "x": x,
                "y": ys,
                "title": final.get("title") or "",
                "columns": cols,
                "rows": last["rows"],
            }
    rec = {
        "question": question,
        "answer": answer,
        "chart": chart,
        "queries": queries,
        "provider": provider.name,
        "model": provider.model,
        "seconds": round(time.perf_counter() - t0, 1),
        "at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    cache.put(key, rec)
    return {**rec, "cached": False}
