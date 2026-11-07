from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lithology.config import load_config
from tests.synth import make_document

REPO = Path(__file__).resolve().parents[1]
HAS_TESSERACT = shutil.which("tesseract") is not None

needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


STANDARD_INTERVALS = [
    (30, "Shale"),
    (20, "Sandstone"),
    (40, "Shale"),
    (10, "Limestone"),
    (25, "Siltstone"),
    (15, "Sandstone"),
    (30, "Shale"),
]


@pytest.fixture(scope="session")
def synth_doc():
    return make_document(STANDARD_INTERVALS)


@pytest.fixture
def synth_dataset(tmp_path, synth_doc):
    """A dataset folder (images/ + bbox_annotation/) holding one synthetic log."""
    images = tmp_path / "images"
    anns = tmp_path / "bbox_annotation"
    images.mkdir()
    anns.mkdir()
    synth_doc.image.save(images / "synthetic_5in_md.tif", compression="tiff_lzw", dpi=(200, 200))
    (anns / "synthetic_5in_md.json").write_text(json.dumps(synth_doc.annotation))
    return tmp_path
