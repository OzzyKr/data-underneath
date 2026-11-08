"""`lith run` flags: box source and the tier-2 guard."""

from __future__ import annotations

import yaml

from lithology import cli


def test_detect_boxes_and_tier_flags_reach_the_worker_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_docs_from", lambda inputs: [])  # stop right after the config is built
    rc = cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o"), "--detect-boxes", "--tier", "1"])
    assert rc == 2  # no documents (by design here)
    forced = yaml.safe_load((tmp_path / "o" / "_config_run_overrides.yaml").read_text())
    assert forced["layout"]["source"] == "detect" and forced["ai"]["tier"] == 1


def test_run_refuses_to_start_when_tier2_is_configured_but_unavailable(tmp_path, monkeypatch):
    import lithology.ai.embed as embed

    monkeypatch.setattr(embed, "availability", lambda: "torch is not installed")
    reached = []
    monkeypatch.setattr(cli, "_docs_from", lambda inputs: reached.append(1) or [])
    assert cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert not reached  # stopped before any work: no silent classical-only run
    assert cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o1"), "--tier", "1"]) == 2
    assert reached  # classical-only on purpose passes the guard


def test_lith_layout_writes_needs_review(tmp_path, monkeypatch, synth_dataset, synth_doc):
    import csv
    import json
    from types import SimpleNamespace

    import lithology.layout
    from lithology.models import DocumentBoxes

    gt = synth_doc.annotation["bounding_boxes"]
    pred = dict(gt, lithology_legend=[0, 0, 5, 5])  # the legend box is wrong (IoU 0) and confidently so
    conf = {"depth_column": 0.3, "interpreted_lithology_column": 0.9, "lithology_legend": 0.8}
    fake = SimpleNamespace(boxes=DocumentBoxes.from_dict(pred), confidence=conf, diagnostics={})
    monkeypatch.setattr(lithology.layout, "detect_boxes", lambda img, cfg: fake)
    out = tmp_path / "lay"
    assert cli.main(["layout", str(synth_dataset / "images"), "--out", str(out), "--no-images"]) == 0
    (row,) = list(csv.DictReader((out / "layout_summary.csv").open()))
    assert row["needs_review"] == "depth_column"
    assert row["conf_depth_column"] == "0.3" and "iou_lithology_legend" in row  # existing columns kept
    page = json.loads((out / "synthetic_5in_md_layout.json").read_text())
    assert page["needs_review"] == ["depth_column"]
    s = json.loads((out / "layout_summary.json").read_text())
    nr = s["needs_review"]
    assert nr["threshold"] == 0.5 and nr["pages"] == 1 and nr["boxes"] == 1
    # depth flagged but right; legend unflagged and wrong (the rule misses a confident mistake); lith right
    want = {"pages": 1, "flagged_wrong": 0, "flagged_right": 1, "unflagged_wrong": 1, "unflagged_right": 1}
    assert nr["dev"] == nr["all"] == want
    assert "page_mean" in s["dev"]["depth_column"]  # existing fields kept
