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
    # Case-insensitive: names are compared that way downstream, and OCR versions differ on capital letters.
    labels = {e.label.lower() for e in entries}
    assert labels == {n.lower() for n in doc.legend_names}, labels


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


@pytest.mark.parametrize(
    ("label", "rock"),
    [
        ("Mud loss", False),
        ("Gas Good", False),
        ("DRILLING SYMBOL", False),
        ("Argillaceous", False),
        ("Sandy", False),
        ("Casing shoe", False),
        ("Silty Sandstone", True),
        ("Sand and Sandstone", True),
        ("Clay-S", True),
        ("SALT", True),
        ("Limestone", True),
        ("Others-H", True),
        ("rite", False),
        ("fio", False),
        ("Casing Top", False),
        ("Pumps Off Gas", False),
        ("Tripping (POOH-RIH)", False),
        ("Oil Moderate", False),
        ("Shell fragments", False),
        ("CHLORITIZATION", False),
        ("NO SAMPLE", False),
        ("Diatoms", False),
        ("Diatomite", True),
        ("Gas sand", True),
        ("Bitumen", True),
        ("QUARTZ LATITE", True),
        ("Silstone", True),
    ],
)
def test_drilling_symbols_and_modifiers_are_not_rock_types(label, rock):
    from lithology.legend import is_rock_type

    assert is_rock_type(label) is rock


# ---------------------------------------------------------------- naming decisions


def _naming_cfg():
    from lithology.config import load_config

    return load_config().naming


def test_naming_agree_named_disagree_unresolved():
    import numpy as np

    from lithology.naming import decide

    cfg = _naming_cfg()
    dist = np.array([0.5, 0.6, 5.0])  # classical unsure between 0 and 1
    agree = decide(dist, np.array([0.90, 0.80, 0.2]), cfg)  # pretrained sure, and the same pick
    assert agree.swatch == 0 and agree.confidence == cfg.confidence_agree
    weak = decide(dist, np.array([0.90, 0.89, 0.2]), cfg)  # both unsure: the same lead is not a decision
    assert weak.swatch is None and weak.candidates[0] == 0
    unsure = decide(dist, np.array([0.89, 0.90, 0.2]), cfg)  # both unsure and they disagree
    assert unsure.swatch is None and set(unsure.candidates) >= {0, 1}
    pretrained_sure = decide(dist, np.array([0.70, 0.95, 0.2]), cfg)
    assert pretrained_sure.swatch == 1


def test_template_match_decisive_and_tie_is_unresolved():
    import numpy as np

    from lithology.naming import decide

    cfg = _naming_cfg()
    dist = np.array([0.1, 3.0, 4.0])  # classical decisively says 0...
    exact = decide(dist, None, cfg, template=np.array([0.40, 0.99, 0.50]))  # ...but pixels say 1, clearly
    assert exact.swatch == 1
    tie = decide(dist, None, cfg, template=np.array([0.99, 1.00, 0.30]))  # pixel-identical to two swatches
    assert tie.swatch is None and set(tie.candidates) == {0, 1}
    no_opinion = decide(dist, None, cfg, template=np.array([0.2, 0.3, 0.1]))  # weak template: classical decides
    assert no_opinion.swatch == 0


def test_template_similarity_finds_identical_pattern_and_respects_colour():
    import numpy as np

    from lithology.naming import template_similarity
    from tests.synth import paint

    def swatch(rock):
        a = np.full((40, 60, 3), 255, np.uint8)
        paint(a, 0, 40, 0, 60, rock)
        return a

    col = np.full((80, 40, 3), 255, np.uint8)
    paint(col, 0, 80, 0, 40, "Shale")
    sims = template_similarity([[col]], [swatch("Shale"), swatch("Sandstone"), swatch("Siltstone")])
    assert sims[0, 0] > 0.95 and sims[0, 0] - max(sims[0, 1], sims[0, 2]) > 0.3


def test_lenient_name_match():
    from lithology.evaluate import lenient_match

    assert lenient_match("Sand and Sandstone", "Sandstone")
    assert lenient_match("SHALE", "Shale")
    assert not lenient_match("Silty Sandstone", "Siltstone")
    assert not lenient_match("Unresolved", "Shale")


def test_unknown_words_are_decided_by_the_swatch_picture():
    import numpy as np

    from lithology.legend import is_rock_type

    fill = np.full((40, 60, 3), 255, np.uint8)
    fill[::6, :, :] = 0  # ruled pattern over the whole box: a rock texture
    glyph = np.full((40, 60, 3), 255, np.uint8)
    glyph[16:24, 26:34] = 0  # one small symbol in the middle
    assert is_rock_type("Porphyry", fill) is True
    assert is_rock_type("Porphyry", glyph) is False


@pytest.mark.parametrize(
    "printed, fixed",
    [
        ("Silstone", "Siltstone"),
        ("Dolornitic", "Dolomitic"),
        ("lirnestone", "limestone"),
        ("Llmestone", "Limestone"),
        ("Dolomites", "Dolomites"),  # a plural is not a misspelling
        ("Sandstones", "Sandstones"),
        ("Shale", "Shale"),
    ],
)
def test_spelling_fixes(printed, fixed):
    from lithology.legend import correct_word

    assert correct_word(printed) == fixed


def test_best_guess_variants_and_identical_swatches():
    import numpy as np

    from lithology.naming import best_guess

    cfg = _naming_cfg()
    labels = ["Sandy clay-shale", "Clay-shale", "Halite", "Clay", "Shale"]
    dist = np.array([0.1, 0.5, 9.0, 3.0, 3.1])
    sim_split = np.array([0.80, 0.90, 0.1, 0.2, 0.2])  # pretrained prefers the plain one: the opinions split
    # opinions split between variants of one rock: the plainest legend entry
    pick, conf, why, flagged = best_guess([0, 1], labels, dist, sim_split, None, [], cfg)
    assert labels[pick] == "Clay-shale" and conf == cfg.confidence_lookalike and "plainest" in why and flagged
    # drawn identically (Clay = Shale) and the opinions split: the more common rock
    pick, _, why, _ = best_guess([3, 4], labels, dist, np.array([0.1, 0.1, 0.1, 0.5, 0.4]), None, [{3, 4}], cfg)
    assert labels[pick] == "Shale" and "identically" in why
    # otherwise the best total rank over all opinions (pretrained and template overrule a weak classical lead)
    pick, conf, _, flagged = best_guess(
        [2, 3],
        labels,
        np.array([0.1, 0.5, 2.0, 2.1, 9.0]),
        np.array([0.1, 0.2, 0.5, 0.9, 0.3]),
        np.array([np.nan, np.nan, 0.6, 0.9, np.nan]),
        [],
        cfg,
    )
    assert labels[pick] == "Clay" and flagged


def test_best_guess_unanimous_and_colour():
    import numpy as np

    from lithology.naming import best_guess

    cfg = _naming_cfg()
    labels = ["SHALE", "SHALE GRAY", "SILTSTONE", "ANHYDRITE"]
    dist = np.array([2.0, 1.0, 5.0, 5.0])
    sim = np.array([0.88, 0.90, 0.5, 0.5])
    tmpl = np.array([0.65, 0.84, 0.0, 0.0])
    # three unsure opinions all lean to SHALE GRAY: named, not overridden by the plain variant
    pick, conf, why, flagged = best_guess([0, 1], labels, dist, sim, tmpl, [], cfg)
    assert labels[pick] == "SHALE GRAY" and not flagged and conf == cfg.confidence_one_sure
    # an orange bed: only orange swatches can be the guess, whatever the colour-blind opinions say
    colour_ok = np.array([False, False, True, True])
    pick, _, _, _ = best_guess([0, 1], labels, dist, sim, None, [], cfg, colour_ok=colour_ok)
    assert labels[pick] in ("SILTSTONE", "ANHYDRITE")


def test_rock_parts():
    from lithology.naming import rock_parts

    assert rock_parts("Sandy clay-shale")[0] == rock_parts("Clay-shale")[0]
    assert rock_parts("Sand")[0] == rock_parts("Sandstone")[0]
    assert rock_parts("Dolomitic limestone") == (frozenset({"limestone"}), 1)
    assert rock_parts("Shale")[0] != rock_parts("Clay")[0]


def test_flagged_beds_are_similar_types_or_unresolved():
    import numpy as np

    from lithology.naming import classify_flagged

    fills = np.array([[80.0, -40, 40], [81.0, -41, 41], [60.0, 10, 60], [79.0, -39, 40]])  # 0, 1, 3 the same green
    # narrowed to look-alike entries of the bed's own colour -> similar types
    assert classify_flagged(0, [1, 2, 3], fills, True, 12.0) == ("similar", [0, 1, 3])
    # candidates that do not look alike -> unresolved
    assert classify_flagged(0, [2], fills, True, 12.0)[0] == "unresolved"
    # the bed does not even have the guess's colour -> unresolved
    assert classify_flagged(0, [1, 3], fills, False, 12.0)[0] == "unresolved"


def test_stacked_swatches_one_pixel_apart_stay_separate():
    """Legends that stack swatches with a 1 px gap (GC297) must give one candidate per swatch, not one tall strip."""
    import numpy as np

    from lithology.legend import _swatch_candidates

    gray = np.full((560, 300), 255, np.uint8)
    for k in range(11):  # 11 filled swatches, 45 px tall, 1 px apart: the column is taller than 45% of the legend
        y = 20 + k * 46
        gray[y : y + 45, 10:77] = 120
    boxes = _swatch_candidates(gray, [], min_px=8, content=gray < 200)
    assert len(boxes) == 11
    assert all(b.height == 45 and b.width == 67 for b in boxes)


def test_snap_to_gap_moves_pattern_edges_into_the_gap_and_survives_column_ends():
    import numpy as np

    from lithology.features import FeatureScaler, describe_ranges, row_stats
    from lithology.segment import snap_to_gap

    col = np.full((200, 40, 3), 255, np.uint8)
    for y in range(0, 100, 20):  # bed A: a dash row every 20 rows
        col[y + 8 : y + 11, 5:35] = 0
    for y in range(110, 200, 6):  # bed B: vertical bars, starting at row 110 (gap 91..109)
        col[y : y + 4, 5:35:6] = 0
    stats = row_stats(col, 110)
    starts = np.arange(0, 180, 20)
    scaler = FeatureScaler.from_blocks(describe_ranges(stats, starts, starts + 20))
    edge = snap_to_gap(stats, scaler, 92, 20, 20)  # refinement left the edge on A's last dash row
    assert 95 <= edge <= 108
    assert snap_to_gap(stats, scaler, 195, 20, 20) == 195  # near the end of the column: unchanged, no crash
