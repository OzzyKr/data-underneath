"""Unit tests per pipeline step, on synthetic images with exact ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from lithology.cluster import cluster_intervals, merge_adjacent
from lithology.evaluate import normalise_name, score_document
from lithology.features import FeatureScaler, describe_ranges, row_stats
from lithology.io import crop_rgb
from lithology.legend import clean_label, correct_word, read_legend
from lithology.models import Box, Interval
from lithology.render import read_csv, render_annotated, write_csv
from lithology.scale import DepthLabel, estimate_scale, fit_line, hypotheses
from lithology.segment import core_box_x, interval_descriptors, segment_column
from tests.conftest import needs_tesseract
from tests.synth import make_document, paint

# ---------------------------------------------------------------- depth scale


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12300", [(12300.0, True), (2300.0, False)]),
        ("12,300,", [(12300.0, True), (2300.0, False)]),  # trailing comma is tick ink
        ("712210", [(12210.0, False)]),  # leading tick read as '7'; 712210 ft is implausible
        ("13239.3", []),  # decimals belong to a secondary (TVD) scale
        ("abc", []),
    ],
)
def test_depth_label_hypotheses(text, expected):
    assert hypotheses(text, 45000) == expected


def test_fit_line_rejects_outliers_and_prefers_printed_reading():
    rng = np.random.default_rng(1)
    labels = []
    for k, depth in enumerate(range(25400, 25800, 20)):  # 20 px/ft, labels every 20 ft
        y = 4400 + (depth - 25400) * 20 + rng.normal(0, 1)
        labels.append(DepthLabel(y, float(depth), 95, k, True))
        labels.append(DepthLabel(y, float(depth - 20000), 95, k, False))  # the dropped-digit hypothesis
    labels.append(DepthLabel(5000.0, 99999.0, 90, 999, True))  # garbage
    labels.append(DepthLabel(6000.0, 5520.0, 90, 998, True))  # a partial read that suits the wrong line
    s, b, inl = fit_line(labels, residual_ft=3.0)
    assert abs(1 / s - 20.0) < 0.05
    assert abs((s * 4400 + b) - 25400) < 0.5
    assert all(lb.primary for lb in inl)


@needs_tesseract
@pytest.mark.parametrize("rotate", [False, True])
def test_estimate_scale_reads_printed_labels(rotate, cfg):
    doc = make_document([(200, "Shale")], rotate_labels=rotate)
    dc = Box.from_seq(doc.annotation["bounding_boxes"]["depth_column"])
    scale, _ = estimate_scale(doc.image, dc, "synthetic", cfg.scale)
    assert scale is not None
    assert abs(scale.px_per_ft - doc.px_per_ft) < 0.05
    top_y = doc.truth[0][0]
    assert abs(scale.depth(top_y) - 5000.0) < 1.0


# ---------------------------------------------------------------- features / segmentation / clustering


def test_descriptors_separate_patterns_and_repeat_exactly():
    arr = np.full((400, 60, 3), 255, np.uint8)
    paint(arr, 0, 100, 0, 60, "Shale")
    paint(arr, 100, 200, 0, 60, "Sandstone")
    paint(arr, 200, 300, 0, 60, "Shale")
    st = row_stats(arr, 110)
    d = describe_ranges(st, np.array([0, 100, 200]), np.array([100, 200, 300]))
    assert np.abs(d[0] - d[2]).max() < 1e-6 + 0.05 * np.abs(d[0]).max()
    assert np.linalg.norm(d[0] - d[1]) > 10


def test_segment_column_finds_true_boundaries(synth_doc, cfg):
    lc = Box.from_seq(synth_doc.annotation["bounding_boxes"]["interpreted_lithology_column"])
    xa, xb = core_box_x(lc.x0, lc.x1, cfg.segment.core_width_fraction)
    col = crop_rgb(synth_doc.image, Box(xa, lc.y0, xb, lc.y1))
    seg = segment_column(col, synth_doc.px_per_ft, cfg.segment)
    found = [lc.y0 + s.top for s in seg.segments[1:]]
    truth = [t[0] for t in synth_doc.truth[1:]]
    tol = 2.0 * synth_doc.px_per_ft
    for t in truth:
        assert min(abs(f - t) for f in found) <= tol, f"missed boundary at {t}: {found}"
    filled = [s for s in seg.segments if not s.blank]
    ids = cluster_intervals(interval_descriptors(seg, filled), seg.scaler, cfg.cluster)
    merged = merge_adjacent([s.top for s in filled], [s.bottom for s in filled], ids.tolist())
    assert len(merged) == len(synth_doc.truth)
    rocks = [t[2] for t in synth_doc.truth]
    cid = {}
    for (_, _, c), rock in zip(merged, rocks, strict=True):  # same rock <-> same cluster, both ways
        assert cid.setdefault(rock, c) == c
    assert len(set(cid.values())) == len(cid)


def test_blank_paper_is_not_an_interval(cfg):
    arr = np.full((600, 40, 3), 255, np.uint8)
    paint(arr, 200, 400, 0, 40, "Shale")
    seg = segment_column(arr, 10.0, cfg.segment)
    filled = [s for s in seg.segments if not s.blank]
    assert len(filled) == 1
    assert abs(filled[0].top - 200) <= 10 and abs(filled[0].bottom - 400) <= 10


def test_merge_adjacent():
    assert merge_adjacent([0, 10, 20, 30], [10, 20, 30, 40], [0, 0, 1, 0]) == [
        (0, 20, 0),
        (20, 30, 1),
        (30, 40, 0),
    ]


def test_scaler_floor_prevents_division_blowup():
    s = FeatureScaler.from_blocks(np.zeros((10, 12)))
    assert np.all(s.scale > 0)


# ---------------------------------------------------------------- legend


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("Llmestone", "Limestone"),
        ("Slltstone", "Siltstone"),
        ("Anhydrlte", "Anhydrite"),
        ("Dolomites", "Dolomites"),
        ("Shale", "Shale"),
        ("Mud", "Mud"),
    ],
)
def test_correct_word_only_fixes_ocr_confusions(raw, fixed):
    assert correct_word(raw) == fixed


def test_clean_label_keeps_codes():
    assert clean_label("  Clay-SCS ") == "Clay-SCS"
    assert clean_label("Sllty  Sandstone") == "Silty Sandstone"


@needs_tesseract
@pytest.mark.parametrize("layout", ["right", "below"])
def test_read_legend_pairs_swatches_with_labels(layout, cfg):
    doc = make_document([(50, "Shale")], legend_layout=layout)
    lg = Box.from_seq(doc.annotation["bounding_boxes"]["lithology_legend"])
    entries = read_legend(crop_rgb(doc.image, lg), (lg.x0, lg.y0), 200.0, cfg.legend)
    labels = {e.label for e in entries}
    assert labels == set(doc.legend_names), labels


# ---------------------------------------------------------------- evaluation + outputs


def _ivs(spec):
    return [Interval(a, b, name, i, a / 10, b / 10) for i, (a, b, name) in enumerate(spec)]


def test_score_perfect_and_shifted():
    truth = _ivs([(0, 100, "Shale"), (100, 200, "Sand"), (200, 300, "Shale")])
    perfect = score_document(truth, truth)
    assert perfect.lithology_accuracy == 1.0 and perfect.boundary_f1 == 1.0 and perfect.depth_mae_ft == 0.0
    shifted = _ivs([(0, 115, "Shale"), (115, 200, "Sand"), (200, 300, "Shale")])
    s = score_document(shifted, truth, tolerance_ft=2.0)  # 15 px = 1.5 ft: still a hit
    assert s.boundary_f1 == 1.0 and s.lithology_accuracy == pytest.approx(0.95)
    far = _ivs([(0, 140, "Shale"), (140, 200, "Sand"), (200, 300, "Shale")])
    assert score_document(far, truth).boundary_recall == 0.5


def test_normalise_name():
    assert normalise_name("Sand-SS") == normalise_name("sand ss")


def test_csv_roundtrip_and_render(tmp_path, synth_doc):
    ivs = _ivs([(420, 500, "Shale"), (500, 640, "Sandstone")])
    write_csv(ivs, tmp_path / "x.csv")
    back = read_csv(tmp_path / "x.csv")
    assert [(b.top_y, b.bottom_y, b.lithology, b.cluster_id) for b in back] == [
        (420, 500, "Shale", 0),
        (500, 640, "Sandstone", 1),
    ]
    from lithology.models import DocumentBoxes

    view = render_annotated(synth_doc.image, DocumentBoxes.from_dict(synth_doc.annotation["bounding_boxes"]), ivs)
    assert view.width > 0 and view.height > 0
