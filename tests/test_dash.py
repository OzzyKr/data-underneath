"""Dashboard back end: titles, rock colours, the database build, the API, the SQL guard, the AI loop, live runs."""

from __future__ import annotations

import json
import time

import duckdb
import pytest

from lithology.dash.ai import Provider
from lithology.dash.insights import AnswerCache, ask, check_sql, snapshot
from lithology.dash.paths import Paths
from lithology.dash.rocks import FAMILY_COLOURS, log_colours, rock_family
from lithology.dash.titles import short_title
from tests.conftest import needs_tesseract

DOC = "synthetic_5in_md"


def test_short_titles():
    assert short_title("MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft") == "MC765 P011 ST00BP00"
    assert short_title("60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400") == "Shenzi SB203 ST00BP01"
    assert short_title("0000000009_d-582_0000020435-0001_log06_fe_5in_md") == "D-582 log06"
    assert short_title("60811405120000_1inch tvd mudlog") == "Well 6081140512"


def test_rock_family_uses_the_head_noun():
    assert rock_family("Sandy Clay-Shale") == "Shale"
    assert rock_family("Dolomitic limestone") == "Limestone"
    assert rock_family("Silty Sandstone") == "Sand"
    assert rock_family("Halite") == "Salt"
    assert rock_family("Casing Top") == "Other"
    assert rock_family("Unresolved") == "Unresolved"


def test_log_colours_follow_the_legend_but_stay_distinct():
    c = log_colours([("Clay-S", (0, 255, 0), 255.0), ("Clay-M", (0, 255, 0), 255.0), ("Shale", (255, 255, 255), 0.0)])
    assert c["Clay-S"] != c["Clay-M"]  # same legend green, still told apart
    assert c["Shale"] == FAMILY_COLOURS["Shale"]  # black-and-white swatch: family colour
    assert c["Clay-S"] == FAMILY_COLOURS["Clay"]  # the legend's green snaps to the validated green


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM logs",
        "SELECT 1; DROP TABLE logs",
        "ATTACH 'x.duckdb'",
        "SELECT * FROM read_csv('secrets.csv')",
        "SELECT * FROM '/etc/passwd.csv'",
        "COPY logs TO 'out.csv'",
        "PRAGMA database_list",
        "SET enable_external_access = true",
    ],
)
def test_sql_guard_rejects_writes_and_files(sql):
    with pytest.raises(ValueError):
        check_sql(sql)


def test_sql_guard_allows_reads():
    assert check_sql("WITH a AS (SELECT 1 AS x) SELECT x FROM a;") == "WITH a AS (SELECT 1 AS x) SELECT x FROM a"
    assert check_sql("select replace(title, 'a', 'b'), used from legend")


# ------------------------------------------------------------------------------------------------ built dashboard


@pytest.fixture(scope="module")
def dash_repo(tmp_path_factory, synth_doc, cfg):
    """A tiny repo: one synthetic scan, run through the pipeline and built into a dashboard."""
    pytest.importorskip("fastapi")
    from lithology.cli import main
    from lithology.dash.store import build

    repo = tmp_path_factory.mktemp("repo")
    images, anns = repo / "dataset" / "images", repo / "dataset" / "bbox_annotation"
    images.mkdir(parents=True)
    anns.mkdir(parents=True)
    synth_doc.image.save(images / f"{DOC}.tif", compression="tiff_lzw", dpi=(200, 200))
    (anns / f"{DOC}.json").write_text(json.dumps(synth_doc.annotation))
    assert main(["run", str(images), "--out", str(repo / "out"), "--workers", "1", "--no-images"]) == 0
    build(Paths(repo), workers=1)
    return repo


@pytest.fixture(scope="module")
def client(dash_repo):
    from fastapi.testclient import TestClient

    from lithology.dash.app import create_app

    with TestClient(create_app(Paths(dash_repo))) as c:
        yield c


@needs_tesseract
def test_api_serves_the_log_and_its_exact_csv(client, dash_repo):
    logs = client.get("/api/logs").json()
    assert [g["doc_id"] for g in logs] == [DOC]
    lid = logs[0]["log_id"]
    detail = client.get(f"/api/logs/{lid}").json()
    assert detail["tiles"]["n"] >= 1 and detail["boxes"]["interpreted_lithology_column"]
    col = detail["tiles"]["col"]
    assert col["n"] >= 1 and col["box"][1] == detail["boxes"]["interpreted_lithology_column"][1]
    assert client.get(f"/tiles/{lid}/c0000.webp").status_code == 200
    page = detail["tiles"]["page"]
    assert page["image"] == [detail["image_w"], detail["image_h"]] and page["n"] >= 1
    assert client.get(f"/tiles/{lid}/p0000.webp").status_code == 200
    assert {r["rock"] for r in detail["rocks"]} >= {"Shale", "Sandstone"}
    assert any(e["used"] and e["swatch_url"] for e in detail["legend"])
    rows = client.get(f"/api/logs/{lid}/intervals").json()
    assert rows and all(r["color"].startswith("#") for r in rows)
    csv = client.get(f"/api/logs/{lid}/csv")
    assert csv.status_code == 200
    assert csv.content == (dash_repo / "out" / f"{DOC}.csv").read_bytes()
    tile = client.get(f"/tiles/{lid}/t0000.webp")
    assert tile.status_code == 200 and tile.content[:4] == b"RIFF"


@needs_tesseract
def test_insights_work_offline(client):
    ins = client.get("/api/insights").json()
    assert ins["headline"]["logs"] == 1 and ins["headline"]["beds"] > 0
    assert ins["mix"][0]["shares"] and abs(sum(s["share"] for s in ins["mix"][0]["shares"]) - 1) < 1e-6
    assert ins["notes"]["mix"]["source"] == "computed" and ins["notes"]["mix"]["text"]
    r = client.post("/api/ask", json={"question": "How many beds are there?"})
    assert r.status_code == 503  # offline and never answered before


class FakeProvider(Provider):
    """Plays the model: one SQL query, then the final answer with a chart."""

    name = "fake"
    model = "fake-1"

    def tool_loop(self, system, user, tools, execute, max_steps=8):
        out, is_err, _ = execute("run_sql", {"sql": "SELECT drop_me FROM logs"})
        assert is_err  # a bad query comes back as an error the model can fix
        out, is_err, _ = execute(
            "run_sql", {"sql": "SELECT rock_family, count(*) AS beds FROM intervals GROUP BY 1 ORDER BY 2 DESC"}
        )
        assert not is_err and "rock_family" in out
        _, _, stop = execute(
            "show_answer", {"text": "Shale has the most beds.", "chart": "bar", "x": "rock_family", "y": ["beds"]}
        )
        assert stop
        return ""


@needs_tesseract
def test_ask_runs_read_only_sql_and_caches(dash_repo, tmp_path):
    import threading

    con = duckdb.connect(str(Paths(dash_repo).db))  # same config as the app fixture's connection
    mem = snapshot(con)
    with pytest.raises(duckdb.Error):  # even past the guard, the copy cannot read files
        mem.execute(f"SELECT * FROM '{dash_repo / 'out' / (DOC + '.csv')}'").fetchall()
    cache = AnswerCache(tmp_path / "answers.json")
    lock = threading.Lock()
    res = ask("Which rock has the most beds?", FakeProvider(), mem, lock, cache)
    assert res["answer"] == "Shale has the most beds."
    assert res["chart"]["type"] == "bar" and res["chart"]["rows"]
    assert res["queries"] == ["SELECT rock_family, count(*) AS beds FROM intervals GROUP BY 1 ORDER BY 2 DESC"]
    offline = ask("which rock has the MOST beds", Provider(), mem, lock, cache)
    assert offline["cached"] and offline["answer"] == res["answer"]


def _wait_run(client, job: str) -> dict:
    deadline = time.time() + 120
    while time.time() < deadline:
        s = client.get(f"/api/runs/{job}").json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.5)
    return s


@needs_tesseract
def test_live_run_adds_a_log(client, dash_repo, synth_doc):
    img = (dash_repo / "dataset" / "images" / f"{DOC}.tif").read_bytes()
    bad = client.post("/api/runs", files=[("files", ("x.tif", img, "image/tiff"))], data={"boxes": "guess"})
    assert bad.status_code == 400  # unknown box mode
    files = [
        ("files", ("live_copy.tif", img, "image/tiff")),
        ("files", ("live_copy.json", json.dumps(synth_doc.annotation).encode(), "application/json")),
    ]
    s = _wait_run(client, client.post("/api/runs", files=files).json()["id"])
    assert s["state"] == "done", s
    assert {"load", "layout", "scale", "legend", "tiles"} <= set(s["steps"]) and s["peak_rss_mb"] > 0
    assert s["boxes"] == {"source": "json"}  # a box file was uploaded: it is used
    logs = client.get("/api/logs").json()
    live = [g for g in logs if g["source"] == "live"]
    assert [g["doc_id"] for g in live] == ["live_copy"] and live[0]["log_id"] == s["log_id"]
    assert client.get(f"/api/logs/{s['log_id']}").json()["boxes_source"] == "json"


@needs_tesseract
def test_live_run_without_box_file_uses_the_layout_detector(client, dash_repo, synth_doc):
    img = (dash_repo / "dataset" / "images" / f"{DOC}.tif").read_bytes()
    s = _wait_run(client, client.post("/api/runs", files=[("files", ("no_boxes.tif", img, "image/tiff"))]).json()["id"])
    assert s["state"] == "done", s
    assert s["boxes"]["source"] == "detector" and s["boxes"]["iou_vs_json"] is None
    detail = client.get(f"/api/logs/{s['log_id']}").json()
    assert detail["boxes_source"] == "detector" and detail["boxes"]["interpreted_lithology_column"]

    # a box file uploaded but the detector forced: the file only scores the detected boxes
    files = [
        ("files", ("forced.tif", img, "image/tiff")),
        ("files", ("forced.json", json.dumps(synth_doc.annotation).encode(), "application/json")),
    ]
    s = _wait_run(client, client.post("/api/runs", files=files, data={"boxes": "detect"}).json()["id"])
    assert s["state"] == "done", s
    assert s["boxes"]["source"] == "detector"
    assert s["boxes"]["iou_vs_json"]["interpreted_lithology_column"] > 0.9
