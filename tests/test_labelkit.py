from __future__ import annotations

import json

import yaml

from lithology.labelkit import choose_window, import_kit
from lithology.models import Interval
from lithology.render import read_csv


def _kit(tmp_path, rows: str):
    kit, labels = tmp_path / "kit", tmp_path / "labels"
    kit.mkdir()
    labels.mkdir()
    (labels / "manifest.yaml").write_text("# comment kept\nsplit: {}\n")
    item = {
        "key": "01_x",
        "doc_id": "doc_x",
        "split": "dev",
        "window_top_ft": 1000.0,
        "window_bottom_ft": 1100.0,
        "knots_y": [0.0, 1000.0],
        "knots_depth": [1000.0, 1100.0],
        "column_y0": 0,
        "column_y1": 1000,
    }
    (kit / "kit.json").write_text(json.dumps([item]))
    (kit / "01_x.csv").write_text("n,top_depth,bottom_depth,lithology,note\n" + rows)
    return kit, labels


def test_import_converts_depths_to_pixel_rows_and_records_split(tmp_path):
    kit, labels = _kit(tmp_path, "1,1000,1020,Shale,\n2,1020,1050,Sandstone,checked\n")
    assert import_kit(kit, labels) == []
    ivs = read_csv(labels / "doc_x.csv")
    assert [(iv.top_y, iv.bottom_y, iv.lithology) for iv in ivs] == [(0, 200, "Shale"), (200, 500, "Sandstone")]
    manifest = (labels / "manifest.yaml").read_text()
    assert manifest.startswith("# comment kept")
    assert yaml.safe_load(manifest)["split"] == {"doc_x": "dev"}


def test_import_reports_gaps_and_bad_rows(tmp_path):
    kit, labels = _kit(tmp_path, "1,1000,1020,Shale,\n2,1025,1050,,\n3,1060,1055,Shale,\n")
    problems = import_kit(kit, labels)
    assert any("does not meet previous bottom" in p for p in problems)
    assert any("missing lithology" in p for p in problems)
    assert any("must be below" in p for p in problems)


def test_choose_window_prefers_variety():
    ivs = [
        Interval(i * 100, (i + 1) * 100, "x", c, 1000 + i * 10, 1010 + i * 10)
        for i, c in enumerate([0, 0, 0, 0, 0, 1, 2, 1, 2, 0])
    ]
    top, bottom = choose_window(ivs, 40.0)
    inside = {iv.cluster_id for iv in ivs if iv.bottom_depth > top and iv.top_depth < bottom}
    assert bottom - top == 40.0 and inside == {0, 1, 2}  # the window shows every rock group
