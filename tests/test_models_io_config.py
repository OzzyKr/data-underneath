from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from lithology.config import load_config
from lithology.io import crop_rgb, depth_reference, discover, family_of, scale_inches_per_100ft
from lithology.models import Box, DepthScale, DocumentBoxes, Interval

# ---------------------------------------------------------------- models


def test_box_iou_and_clip():
    a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
    assert a.iou(a) == 1.0
    assert math.isclose(a.iou(b), 25 / 175)
    assert Box(-5, -5, 20, 20).clip(10, 10) == Box(0, 0, 10, 10)
    with pytest.raises(ValueError):
        Box(5, 5, 5, 10)


def test_depth_scale_linear_and_piecewise():
    s = DepthScale.linear(0.1, 1000.0, "test")
    assert math.isclose(s.depth(0), 1000.0)
    assert math.isclose(s.depth(500), 1050.0)
    assert math.isclose(s.px_per_ft, 10.0)
    p = DepthScale((0.0, 100.0, 300.0), (0.0, 10.0, 50.0), "pw")  # slope changes at y=100
    assert math.isclose(p.depth(50), 5.0)
    assert math.isclose(p.depth(200), 30.0)
    assert math.isclose(p.depth(400), 70.0)  # linear extrapolation of the last segment
    with pytest.raises(ValueError):
        DepthScale((0.0, 0.0), (1.0, 2.0), "bad")


def test_interval_contract_and_csv_row():
    iv = Interval(top_y=10, bottom_y=20, lithology="Shale", cluster_id=0, top_depth=100.0, bottom_depth=101.0)
    assert iv.csv_row() == {
        "top_depth": 100.0,
        "bottom_depth": 101.0,
        "lithology": "Shale",
        "top_y": 10,
        "bottom_y": 20,
        "cluster_id": 0,
    }
    with pytest.raises(ValueError):
        Interval(top_y=20, bottom_y=20)


def test_document_boxes_roundtrip():
    d = {
        "depth_column": [1, 2, 3, 4],
        "interpreted_lithology_column": [5, 6, 7, 8],
        "lithology_legend": [0, 0, 9, 9],
    }
    assert DocumentBoxes.from_dict(d).to_dict() == d
    with pytest.raises(ValueError):
        DocumentBoxes.from_dict({"depth_column": [1, 2, 3, 4]})


# ---------------------------------------------------------------- filename metadata (real dataset names)


@pytest.mark.parametrize(
    ("name", "inches"),
    [
        ("0000000009_d-582_0000020435-0001_log01_fe_1in_md", 1.0),
        ("MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft", 5.0),
        ("60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft", 5.0),
        ("60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872", 1.0),
        ("60811405800500_608114058005_g16786_001_03_00_tvd5_mud", 5.0),
        ("60817410140100_mc822#10bp01 1-240 tvd mud log", 5.0),
        ("60817410140000_mc822#10 1-1200 mud log", 1.0),
        ("60808400140000_hadrian 2_masterlog_2in_10000-21962", 2.0),
        ("Rio_Lavaca_1748_Unit_1_241_MUD", None),
    ],
)
def test_scale_from_filename(name, inches):
    assert scale_inches_per_100ft(name) == inches


def test_family_and_reference():
    assert family_of("60811405800100_mdn-1") == family_of("60811405800300_mdn-1_st01")
    assert family_of("MC765_P011_ST00BP00_1inchMD") == family_of("MC765_P011_ST00BP01_5inchMD")
    assert family_of("Rio_Lavaca_1") != family_of("MC765_P011")
    assert depth_reference("x_5inchTVD_y") == "TVD"
    assert depth_reference("x_1in_md") == "MD"


def test_discover_skips_duplicates_and_non_images(tmp_path):
    images = tmp_path / "images"
    anns = tmp_path / "bbox_annotation"
    images.mkdir()
    anns.mkdir()
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.tif")
    (images / "a(1).tif").write_bytes((images / "a.tif").read_bytes())
    Image.new("RGB", (10, 10), (9, 9, 9)).save(images / "b.tiff")
    (images / "Thumbs.db").write_bytes(b"x")
    (anns / "a.json").write_text("{}")
    disc = discover(images)
    assert sorted(d.doc_id for d in disc.documents) == ["a", "b"]
    reasons = {p.name: why for p, why in disc.skipped}
    assert "duplicate" in reasons["a(1).tif"]
    assert "Thumbs.db" in reasons


def test_discover_finds_box_files_under_common_folder_names(tmp_path):
    for folder in ("annotations", "images"):
        root = tmp_path / folder
        images = root / "images"
        images.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(images / "a.tif")
        anns = root / folder if folder != "images" else images
        anns.mkdir(exist_ok=True)
        (anns / "a.json").write_text("{}")
        assert discover(images).documents[0].annotation_path == anns / "a.json"


def test_crop_rgb_matches_numpy_and_handles_1bit():
    arr = (np.random.default_rng(0).random((500, 40, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    box = Box(5, 37, 33, 433)
    np.testing.assert_array_equal(crop_rgb(img, box, strip=64), arr[37:433, 5:33])
    small = crop_rgb(img, box, scale=0.5)
    assert small.shape == (198, 14, 3)
    bw = Image.fromarray(arr[..., 0] > 128).convert("1")
    assert crop_rgb(bw, box).shape == (396, 28, 3)


# ---------------------------------------------------------------- config


def test_config_loads_hashes_and_validates():
    a, b = load_config(), load_config()
    assert a.hash() == b.hash()
    c = load_config(overrides={"cluster": {"distance_threshold": 9.9}})
    assert c.cluster.distance_threshold == 9.9 and c.hash() != a.hash()
    with pytest.raises(ValueError, match="unknown"):
        load_config(overrides={"cluster": {"nope": 1}})
    with pytest.raises(ValueError, match="integer"):
        load_config(overrides={"runtime": {"workers": 1.5}})
