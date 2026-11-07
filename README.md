# data-underneath

Automated lithology extraction from scanned well-log images, built for the Viridien HackTheBurgh challenge
"Decoding the Subsurface". For each log it finds lithology intervals, groups intervals with the same pattern,
names them from the log's own legend, converts pixel rows to depth, and draws an annotated image.

The default pipeline is deterministic, runs offline on a laptop CPU, and uses classical image processing plus OCR.
No model is trained on the development images.

## Quick start

```bash
brew install tesseract                 # OCR engine (Linux: apt-get install tesseract-ocr)
conda env create -f environment.yml    # Python 3.12 env named "lithology"
conda activate lithology

lith check-data                        # audit dataset/, write data/manifest.json
lith run dataset/images --out out      # CSV + annotated PNG per log, plus run_summary.csv
lith eval --pred out --labels labels   # score against hand labels (dev split), append to eval/results_log.csv
make check                             # lint + tests
```

## Data setup

The dataset is not tracked in git. Place it in `dataset/` at the repo root:

```
dataset/
├── bbox_annotation/   # boxes for depth column, lithology column, legend (.json), one per image
├── bbox_vis/          # organiser previews (.png)
└── images/            # source log images (.tif / .tiff)
```

Images and annotations share a base name. Byte-identical duplicates and non-image files (e.g. `Thumbs.db`) are
skipped automatically. `example/` and `example_notebook.ipynb` are local-only too; when present, a golden
regression test runs against `example/example.csv`.

## Outputs (per document, in `--out`)

| File | Content |
| --- | --- |
| `<doc>.csv` | Submission format: `top_depth, bottom_depth, lithology, top_y, bottom_y, cluster_id` |
| `diagnostics/<doc>.csv` | Same rows plus `confidence` and `notes` (failed physical checks) |
| `<doc>.png` | Annotated lithology column: intervals, boundary depths, labels, colour key |
| `run_summary.csv`, `run_meta.json` | Per-document runtime, peak memory, warnings; commit and config hash |

## How it works

| Step | Module | Method |
| --- | --- | --- |
| 1. Load | `io.py` | Lazy TIFF open, strip-wise crop of only the needed regions |
| 2. Depth scale | `scale.py` | Tesseract on windows of the depth column, all 3 text orientations; exhaustive RANSAC line fit with "dropped digit" hypotheses; cross-checked against filename scale x TIFF DPI |
| 3. Segment | `segment.py`, `features.py` | Per-block colour + pattern descriptors, change points (`ruptures` KernelCPD), boundary refinement to the pixel row, blank-paper detection |
| 4. Cluster | `cluster.py` | Agglomerative clustering with a distance threshold (no fixed k) |
| 5. Legend | `legend.py` | Swatch detection, OCR labels (right-of or below layouts), stroke-width-invariant match descriptors |
| 6. Checks | `checks.py` | Contiguity, containment, depth increasing, minimum thickness; failures lower confidence |
| 7. Output | `render.py` | CSV + annotated image (drawing code ported from the organisers' notebook) |

Every threshold lives in `config/default.yaml`, in feet or as a fraction of a box, never raw pixels.

## Engineering rules

- Same input + same config = same output. No network calls in the default run.
- Every change is scored; results are logged with git commit, config hash and dataset hash.
- Accuracy is claimed only on a frozen holdout (see `labels/manifest.yaml`), split by document family.
- A failed step degrades and is reported; a batch never stops on one bad document.
- Each document runs in its own process, so runtime and peak memory are measured per document.
