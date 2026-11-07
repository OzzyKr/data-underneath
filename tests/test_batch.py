"""The batch runner survives a worker that dies (as when the OS kills it for running out of memory)."""

import os
import time

from lithology.cli import run_batch
from lithology.io import Document


def _job(args):
    doc, mode = args
    if mode == "die":
        os._exit(137)  # what an out-of-memory kill looks like to the pool
    t0 = time.time()
    time.sleep(0.3)
    return {"doc_id": doc.doc_id, "ok": True, "error": "", "t0": t0, "t1": time.time()}


def _doc(name: str) -> Document:
    return Document(name, None, None)  # type: ignore[arg-type]


def test_batch_survives_dead_worker():
    jobs = [(_doc("a"), "ok"), (_doc("b"), "die"), (_doc("c"), "ok"), (_doc("d"), "ok")]
    rows = run_batch(jobs, [100.0] * 4, workers=2, budget_mb=1000.0, fn=_job)
    by = {r["doc_id"]: r for r in rows}
    assert [r["doc_id"] for r in rows] == ["a", "b", "c", "d"]  # input order kept
    assert by["a"]["ok"] and by["c"]["ok"] and by["d"]["ok"]
    assert not by["b"]["ok"] and "died" in by["b"]["error"]


def test_budget_runs_big_job_alone():
    rows = run_batch(
        [(_doc("big"), "ok"), (_doc("s1"), "ok"), (_doc("s2"), "ok")],
        [5000.0, 100.0, 100.0],
        workers=3,
        budget_mb=4000.0,
        fn=_job,
    )
    assert all(r["ok"] for r in rows)
    big = rows[0]
    for small in rows[1:]:  # over the budget: it never shares the machine
        assert small["t1"] <= big["t0"] or small["t0"] >= big["t1"]
    assert not (rows[2]["t1"] <= rows[1]["t0"] or rows[2]["t0"] >= rows[1]["t1"])  # small ones run together
