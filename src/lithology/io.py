"""Dataset discovery, filename metadata and memory-conscious image access."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lithology.models import Annotation, Box

Image.MAX_IMAGE_PIXELS = None  # the logs are legitimately huge (up to ~314k px tall)

IMAGE_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class Document:
    """One log image and (optionally) its annotation file."""

    doc_id: str  # file stem, shared by image, annotation and outputs
    image_path: Path
    annotation_path: Path | None

    @property
    def family(self) -> str:
        return family_of(self.doc_id)

    def annotation(self) -> Annotation | None:
        return Annotation.from_json(self.annotation_path) if self.annotation_path else None


@dataclass(frozen=True)
class Discovery:
    documents: list[Document]
    skipped: list[tuple[Path, str]]  # (path, reason)


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


ANNOTATION_DIRS = ("bbox_annotation", "annotations")  # sibling folders of images/ that may hold the box files


def find_annotation(image: Path, annotations_dir: Path | None = None) -> Path | None:
    """The box file of an image: in ``annotations_dir`` if given, else a known sibling folder or beside the image."""
    dirs = [Path(annotations_dir)] if annotations_dir else [image.parent.parent / d for d in ANNOTATION_DIRS]
    for d in [*dirs, image.parent]:
        a = d / f"{image.stem}.json"
        if a.exists():
            return a
    return None


def discover(images_dir: Path, annotations_dir: Path | None = None) -> Discovery:
    """List log images, pair them with annotations, and skip non-images and byte-identical duplicates.

    Duplicates are found by size first and confirmed by SHA-256, so only same-size files are hashed.
    Of a duplicate group, the copy that has an annotation (then the shortest name) is kept.
    """
    images_dir = Path(images_dir)
    skipped: list[tuple[Path, str]] = []
    candidates: list[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(p)
        elif p.is_file():
            skipped.append((p, "not a TIFF image"))

    def ann_for(p: Path) -> Path | None:
        return find_annotation(p, annotations_dir)

    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in candidates:
        by_size[p.stat().st_size].append(p)
    keep: set[Path] = set(candidates)
    for group in by_size.values():
        if len(group) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for p in group:
            by_hash[_sha256(p)].append(p)
        for same in by_hash.values():
            if len(same) < 2:
                continue
            best = sorted(same, key=lambda q: (ann_for(q) is None, len(q.name), q.name))[0]
            for q in same:
                if q != best:
                    keep.discard(q)
                    skipped.append((q, f"byte-identical duplicate of {best.name}"))

    docs = [Document(p.stem, p, ann_for(p)) for p in candidates if p in keep]
    return Discovery(documents=docs, skipped=skipped)


# ---------------------------------------------------------------------------------------------------------
# Filename metadata (used only as an independent cross-check, never to branch on a specific document)

_FAMILY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(\d{10})\d{4}"), r"api:\1"),  # 14-digit API number: first 10 digits identify the well
    (re.compile(r"^(0{4,}\d+)_"), r"series:\1"),
    (re.compile(r"^([A-Za-z]{2}\d{3})_", re.I), r"block:\1"),  # e.g. MC765_
)


def family_of(doc_id: str) -> str:
    """Group documents that share a well or a layout, for leave-one-family-out validation."""
    for pattern, repl in _FAMILY_RULES:
        m = pattern.match(doc_id)
        if m:
            return m.expand(repl).lower()
    token = re.split(r"[_\s\-]", doc_id, maxsplit=1)[0]
    return f"name:{token.lower()}"


_SCALE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\d)([125])\s*-?\s*in(?:ch)?(?![a-z])", re.I), "in"),  # 5in, 5 in, 5inch, 1inch
    (re.compile(r"(?<!\d)([125])\s*in(?:ch)?(?=md|tvd)", re.I), "in"),  # 5inchMD
    (re.compile(r"[_\s]([125])(?:md|tvd)[_\s]", re.I), "in"),  # _1md_, _5md_
    (re.compile(r"(?:md|tvd)([125])[_\s]", re.I), "in"),  # md5_, tvd5_
    (re.compile(r"(?<!\d)1-(240|480|600|1200)(?!\d)"), "ratio"),  # 1-240 = 1:240
)


def scale_inches_per_100ft(doc_id: str) -> float | None:
    """Vertical log scale from the filename, in inches of paper per 100 ft of depth, or None."""
    for pattern, kind in _SCALE_PATTERNS:
        m = pattern.search(doc_id)
        if m:
            v = float(m.group(1))
            return 1200.0 / v if kind == "ratio" else v
    return None


def depth_reference(doc_id: str) -> str | None:
    low = doc_id.lower()
    if "tvd" in low:
        return "TVD"
    if re.search(r"(?<![a-z])md(?![a-z])|md\d|\dmd|inchmd|in_md|_md", low):
        return "MD"
    return None


# ---------------------------------------------------------------------------------------------------------
# Image access


def open_image(path: Path) -> Image.Image:
    """Open lazily: only the header is read until pixels are requested."""
    return Image.open(path)


def image_dpi(img: Image.Image) -> float | None:
    dpi = img.info.get("dpi")
    if not dpi:
        return None
    y = float(dpi[1] if isinstance(dpi, tuple | list) else dpi)
    return y if y > 1 else None


def crop_rgb(img: Image.Image, box: Box, scale: float = 1.0, strip: int = 2000) -> np.ndarray:
    """Crop ``box`` as an RGB uint8 array (H, W, 3), optionally shrunk by ``scale``.

    Converted (and resized) ``strip`` rows at a time, so a tall 1-bit crop never exists at full size as RGB.
    Ported from the organisers' example notebook.
    """
    box = box.clip(*img.size)
    out_w = max(1, round(box.width * scale))
    out_h = max(1, round(box.height * scale))
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    for ys in range(box.y0, box.y1, strip):
        ye = min(ys + strip, box.y1)
        part = img.crop((box.x0, ys, box.x1, ye)).convert("RGB")
        top = round((ys - box.y0) * scale)
        bottom = max(top + 1, round((ye - box.y0) * scale))
        bottom = min(bottom, out_h)
        if top >= out_h:
            break
        if scale != 1.0:
            part = part.resize((out_w, bottom - top), Image.Resampling.BOX)
        out[top:bottom] = np.asarray(part)[: bottom - top]
    return out
