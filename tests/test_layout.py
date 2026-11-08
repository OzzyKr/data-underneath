"""Step 1, layout detection: find the three boxes without the annotation file."""

from __future__ import annotations

import pytest

from lithology.config import load_config
from lithology.io import discover
from lithology.layout import detect_boxes
from lithology.models import DocumentBoxes
from lithology.pipeline import process
from lithology.scale import estimate_scale
from tests.conftest import STANDARD_INTERVALS, needs_tesseract
from tests.synth import make_document


def _iou(res, gt: DocumentBoxes) -> dict[str, float]:
    return {
        "depth": res.boxes.depth_column.iou(gt.depth_column),
        "lith": res.boxes.lithology_column.iou(gt.lithology_column),
        "legend": res.boxes.legend.iou(gt.legend),
    }


@needs_tesseract
@pytest.mark.parametrize(
    "kw", [{}, {"rotate_labels": True}, {"legend_layout": "below"}], ids=["plain", "rotated-labels", "legend-below"]
)
def test_detects_three_boxes_on_synthetic_logs(kw, cfg):
    doc = make_document(STANDARD_INTERVALS, **kw)
    doc.image.info["dpi"] = (200, 200)
    gt = DocumentBoxes.from_dict(doc.annotation["bounding_boxes"])
    res = detect_boxes(doc.image, cfg.layout)
    iou = _iou(res, gt)
    assert iou["lith"] > 0.9, iou
    assert iou["legend"] > 0.4, iou  # the synthetic legend box is drawn larger than its content
    # the depth box must hold the printed labels: the scale read from it matches the true scale
    scale, _ = estimate_scale(doc.image, res.boxes.depth_column, "synthetic_5in_md", cfg.scale)
    assert scale is not None
    assert abs(scale.px_per_ft - doc.px_per_ft) / doc.px_per_ft < 0.02
    assert all(0.0 <= v <= 1.0 for v in res.confidence.values())


@needs_tesseract
def test_pipeline_step1_detects_and_reports_iou(synth_dataset):
    cfg = load_config(overrides={"layout": {"source": "detect"}})
    doc = discover(synth_dataset / "images").documents[0]
    res = process(doc, cfg)
    assert res.ok, res.error
    assert res.layout["source"] == "detected"
    assert res.layout["iou_vs_annotation"]["interpreted_lithology_column"] > 0.9


@needs_tesseract
def test_pipeline_can_use_annotation_for_ab_comparison(synth_dataset, synth_doc):
    cfg = load_config(overrides={"layout": {"source": "annotation"}})
    doc = discover(synth_dataset / "images").documents[0]
    res = process(doc, cfg)
    assert res.ok, res.error
    assert res.layout["source"] == "annotation"
    assert res.boxes == DocumentBoxes.from_dict(synth_doc.annotation["bounding_boxes"])


def test_layout_source_must_be_known():
    cfg = load_config()
    assert cfg.layout.source in ("auto", "detect", "annotation")


@needs_tesseract
def test_default_uses_the_annotation_and_detects_only_without_one(synth_dataset, synth_doc, tmp_path):
    """The main task must never depend on the detector when the organisers' boxes exist."""
    import shutil

    cfg = load_config()
    assert cfg.layout.source == "auto"
    doc = discover(synth_dataset / "images").documents[0]
    res = process(doc, cfg)
    assert res.ok and res.layout["source"] == "annotation"
    assert res.boxes == DocumentBoxes.from_dict(synth_doc.annotation["bounding_boxes"])
    lone = tmp_path / "no_annotation" / "images"  # the same image with no annotation file next to it
    lone.mkdir(parents=True)
    shutil.copy(doc.image_path, lone / doc.image_path.name)
    res2 = process(discover(lone).documents[0], cfg)
    assert res2.ok and res2.layout["source"] == "detected"


def _with_cuttings_track(doc):
    """Add a cuttings-percent track (two rocks side by side in every row, 30/70) right of the lithology column,
    under a '% Lithology' header: the detector must still choose the interpreted column."""
    import numpy as np
    from PIL import ImageDraw, ImageFont

    from tests.synth import paint

    arr = np.asarray(doc.image).copy()
    b = doc.annotation["bounding_boxes"]["interpreted_lithology_column"]
    y0, y1 = b[1], b[3]
    x0, x1 = 640, 720
    split = x0 + int(0.3 * (x1 - x0))
    paint(arr, y0, y1, x0, split, "Sandstone")
    paint(arr, y0, y1, split, x1, "Shale")
    arr[y0:y1, x0] = 0
    arr[y0:y1, x1 - 1] = 0
    img = __import__("PIL.Image", fromlist=["Image"]).fromarray(arr)
    ImageDraw.Draw(img).text((x0, y0 - 30), "% Lithology", fill=(0, 0, 0), font=ImageFont.load_default(size=18))
    img.info["dpi"] = (200, 200)
    return img


@needs_tesseract
def test_cuttings_percent_track_is_not_chosen(cfg):
    doc = make_document(STANDARD_INTERVALS)
    img = _with_cuttings_track(doc)
    gt = DocumentBoxes.from_dict(doc.annotation["bounding_boxes"])
    res = detect_boxes(img, cfg.layout)
    assert res.boxes.lithology_column.iou(gt.lithology_column) > 0.9


@needs_tesseract
def test_body_found_when_log_ends_early(cfg):
    import numpy as np
    from PIL import Image

    doc = make_document(STANDARD_INTERVALS)
    arr = np.asarray(doc.image)
    tail = np.full((int(0.6 * arr.shape[0]), arr.shape[1], 3), 255, np.uint8)  # long blank tail below the log
    img = Image.fromarray(np.concatenate([arr, tail]))
    img.info["dpi"] = (200, 200)
    gt = DocumentBoxes.from_dict(doc.annotation["bounding_boxes"])
    res = detect_boxes(img, cfg.layout)
    assert res.boxes.lithology_column.iou(gt.lithology_column) > 0.9


def _hatched_page():
    """A colour log whose interpreted column carries a dense black anti-aliased hatch over its fills, between a
    constant one-colour strip and a cuttings-like track of two rocks side by side in every row."""
    import cv2
    import numpy as np
    from PIL import Image

    from tests.synth import ROCKS

    rocks = ["Shale", "Sandstone", "Limestone", "Shale", "Sandstone", "Limestone", "Shale"]
    doc = make_document([(25, r) for r in rocks])
    arr = np.asarray(doc.image).copy()
    b = doc.annotation["bounding_boxes"]["interpreted_lithology_column"]
    x0, y0, x1, y1 = b
    edges = np.linspace(y0, y1, len(rocks) + 1).astype(int)
    for (ya, yb), rock in zip(zip(edges[:-1], edges[1:], strict=True), rocks, strict=True):
        arr[ya:yb, x0:x1] = ROCKS[rock][0]
    # dense diagonal hatch (period 8 px, about 1/3 of the pixels dark or anti-aliased): a blurred line mask in black
    ys, xs = np.mgrid[y0:y1, x0:x1]
    alpha = cv2.GaussianBlur(((xs + ys) % 8 == 0).astype(np.float32), (0, 0), 0.6) * 2.0
    alpha = np.clip(alpha, 0, 1)[..., None]
    arr[y0:y1, x0:x1] = (arr[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)
    # a constant one-colour strip (a legend colour, no pattern) right of the depth column
    arr[y0:y1, 440:470] = ROCKS["Shale"][0]
    # a cuttings-like track: two rocks side by side in every row, proportions and rocks changing down the page
    for k, (ya, yb) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        split = 640 + (20 + 15 * (k % 4))
        arr[ya:yb, 640:split] = ROCKS[rocks[k]][0]
        arr[ya:yb, split:720] = ROCKS[rocks[(k + 1) % len(rocks)]][0]
    for x in (x0, x1 - 1, 440, 469, 640, 719):  # track rules
        arr[y0:y1, x] = 0
    img = Image.fromarray(arr)
    img.info["dpi"] = (200, 200)
    return img, DocumentBoxes.from_dict(doc.annotation["bounding_boxes"])


@needs_tesseract
def test_hatched_column_beats_constant_strip_and_cuttings_track(cfg):
    """The hatch's anti-aliased fringe is not a second colour (it snaps to the fill's legend colour), and a strip of
    one constant colour carries no lithology: the hatched interpreted column must be chosen."""
    img, gt = _hatched_page()
    res = detect_boxes(img, cfg.layout)
    assert res.boxes.lithology_column.iou(gt.lithology_column) > 0.9, res.boxes.lithology_column


@needs_tesseract
@pytest.mark.parametrize("rock", ["Shale", "Sandstone"])
def test_single_rock_log_is_still_chosen(rock, cfg):
    """The constant-strip penalty is soft: a log of one lithology all the way down must still be found."""
    doc = make_document([(150, rock)])
    doc.image.info["dpi"] = (200, 200)
    gt = DocumentBoxes.from_dict(doc.annotation["bounding_boxes"])
    res = detect_boxes(doc.image, cfg.layout)
    assert res.boxes.lithology_column.iou(gt.lithology_column) > 0.9


# ------------------------------------------------------------------------------------------- the needs-review rule

LOW = {"depth_column": 0.3, "interpreted_lithology_column": 0.9, "lithology_legend": 0.45}


def _fake_detector(monkeypatch, boxes: DocumentBoxes, confidence: dict[str, float]) -> list:
    """Replace the detector with one returning fixed boxes and confidences; returns the list of calls."""
    from types import SimpleNamespace

    import lithology.layout

    calls: list = []

    def fake(img, cfg):
        calls.append(cfg)
        return SimpleNamespace(boxes=boxes, confidence=dict(confidence), diagnostics={})

    monkeypatch.setattr(lithology.layout, "detect_boxes", fake)
    return calls


def test_layout_step_flags_boxes_below_the_review_threshold(monkeypatch, synth_doc, tmp_path):
    from lithology.io import Document
    from lithology.pipeline import _layout_step

    gt = DocumentBoxes.from_dict(synth_doc.annotation["bounding_boxes"])
    _fake_detector(monkeypatch, gt, LOW)
    doc = Document("lone", tmp_path / "lone.tif", None)  # no annotation: the boxes are detected
    cfg = load_config()
    _, info = _layout_step(doc, None, cfg, None)
    assert info["source"] == "detected"
    assert info["review_threshold"] == cfg.layout.review_threshold == 0.5
    assert info["needs_review"] == ["depth_column", "lithology_legend"]
    assert info["confidence"] == LOW
    _, info = _layout_step(doc, None, load_config(overrides={"layout": {"review_threshold": 0.2}}), None)
    assert info["needs_review"] == []  # the threshold comes from the config


@needs_tesseract
def test_pipeline_records_and_warns_about_boxes_needing_review(monkeypatch, synth_dataset, synth_doc, tmp_path, caplog):
    import json
    import logging
    import shutil

    from lithology.pipeline import write_outputs

    gt = DocumentBoxes.from_dict(synth_doc.annotation["bounding_boxes"])
    _fake_detector(monkeypatch, gt, LOW)
    src = discover(synth_dataset / "images").documents[0]
    lone = tmp_path / "no_annotation" / "images"
    lone.mkdir(parents=True)
    shutil.copy(src.image_path, lone / src.image_path.name)
    doc = discover(lone).documents[0]
    cfg = load_config()
    with caplog.at_level(logging.WARNING, logger="lithology.pipeline"):
        res = process(doc, cfg)
    assert res.ok, res.error
    assert res.layout["needs_review"] == ["depth_column", "lithology_legend"]
    assert any("needs review" in w and "depth_column" in w for w in res.warnings)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and doc.doc_id in r.getMessage()]
    assert warned and "lithology_legend" in warned[0].getMessage()
    out = tmp_path / "out"
    write_outputs(doc, res, out, cfg, image=False)
    meta = json.loads((out / "diagnostics" / f"{doc.doc_id}_layout.json").read_text())
    assert meta["needs_review"] == ["depth_column", "lithology_legend"] and meta["confidence"] == LOW
    assert (out / f"{doc.doc_id}.csv").read_text().splitlines()[0] == (
        "top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id"
    )  # the interval CSV is unchanged


@needs_tesseract
def test_annotation_runs_are_not_flagged(monkeypatch, synth_dataset):
    calls = _fake_detector(monkeypatch, DocumentBoxes.from_dict({k: [0, 0, 1, 1] for k in LOW}), LOW)
    res = process(discover(synth_dataset / "images").documents[0], load_config())
    assert res.ok, res.error
    assert not calls  # the annotation was used: the detector never ran
    assert res.layout == {"source": "annotation"}
    assert not any("needs review" in w for w in res.warnings)
