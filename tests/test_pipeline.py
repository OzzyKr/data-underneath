"""End-to-end tests: the full pipeline on a synthetic log, and a golden regression test on the organisers' example."""

from __future__ import annotations

from pathlib import Path

import pytest

from lithology.cli import main
from lithology.evaluate import score_document
from lithology.io import Document, discover
from lithology.models import Interval
from lithology.pipeline import process
from lithology.render import read_csv
from tests.conftest import REPO, needs_tesseract


@needs_tesseract
def test_full_pipeline_on_synthetic_log(synth_dataset, synth_doc, cfg):
    doc = discover(synth_dataset / "images").documents[0]
    res = process(doc, cfg)
    assert res.ok, res.error
    truth = [Interval(a, b, rock, None, d0, d1) for a, b, rock, d0, d1 in synth_doc.truth]
    s = score_document(res.intervals, truth)
    assert s.boundary_f1 == 1.0
    assert s.depth_mae_ft < 1.0
    assert s.cluster_ari > 0.99
    assert s.lithology_accuracy > 0.95  # names come from the synthetic legend


@needs_tesseract
def test_cli_run_writes_outputs(synth_dataset, tmp_path):
    out = tmp_path / "out"
    code = main(["run", str(synth_dataset / "images"), "--out", str(out), "--workers", "1"])
    assert code == 0
    assert (out / "synthetic_5in_md.csv").exists()
    assert (out / "synthetic_5in_md.png").exists()
    assert (out / "run_summary.csv").exists() and (out / "run_meta.json").exists()


def test_pipeline_never_raises_on_bad_input(tmp_path, cfg):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not an image")
    res = process(Document("broken", bad, None), cfg)
    assert not res.ok and res.error


EXAMPLE = REPO / "example"
EXAMPLE_ID = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"


@pytest.mark.dataset
@needs_tesseract
@pytest.mark.skipif(not (EXAMPLE / "example.csv").exists(), reason="organisers' example/ folder not present")
def test_golden_example_regression(cfg):
    """Regression guard on the only official ground truth: boundaries and depths must not get worse."""
    doc = Document(EXAMPLE_ID, EXAMPLE / f"{EXAMPLE_ID}.tif", EXAMPLE / f"{EXAMPLE_ID}.json")
    res = process(doc, cfg)
    assert res.ok, res.error
    s = score_document(res.intervals, read_csv(Path(EXAMPLE / "example.csv")))
    assert s.depth_mae_ft < 0.5
    assert s.boundary_f1 >= 0.85
    assert s.cluster_ari >= 0.75


@needs_tesseract
def test_step_hook_and_legend_json(synth_dataset, synth_doc, cfg, tmp_path):
    """Live runs need step progress; the dashboard needs each swatch's fill colour."""
    import json

    from lithology.pipeline import write_outputs

    doc = discover(synth_dataset / "images").documents[0]
    steps: list[str] = []
    res = process(doc, cfg, on_step=steps.append)
    assert res.ok, res.error
    assert steps == ["load", "layout", "scale", "segment", "cluster", "legend", "assemble"]
    write_outputs(doc, res, tmp_path, cfg, image=False)
    legend = json.loads((tmp_path / "diagnostics" / f"{doc.doc_id}_legend.json").read_text())
    by_label = {e["label"]: e for e in legend}
    sand = by_label["Sandstone"]
    assert sand["chroma"] > 100 and sand["rgb"][2] < 120  # yellow fill, not the black dots
    assert by_label["Siltstone"]["chroma"] < 20  # grey swatch


def test_step_hook_errors_do_not_fail_the_document(synth_dataset, cfg):
    def boom(_name: str) -> None:
        raise RuntimeError("progress reporter broke")

    doc = discover(synth_dataset / "images").documents[0]
    res = process(doc, cfg, on_step=boom)
    assert res.error is None or "progress" not in res.error
