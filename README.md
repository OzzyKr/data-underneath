# data-underneath

Automated lithology extraction from scanned mud-log images, built for the Viridien HackTheBurgh challenge
*Decoding the Subsurface*.

For every log the pipeline finds the lithology intervals, groups intervals drawn with the same pattern, names each
group from the log's own legend, converts pixel rows to depth, and writes a CSV plus an annotated image. It is
deterministic, runs offline on a laptop CPU, and trains nothing on the data: classical image processing and OCR
do the work, with a frozen ImageNet ResNet-18 as a second opinion only where the classical methods disagree.

A local dashboard shows the results, runs new scans and lets you question all logs as one dataset.

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Placing the data](#placing-the-data)
4. [Running the pipeline](#running-the-pipeline)
5. [Outputs](#outputs)
6. [Finding the three regions](#finding-the-three-regions)
7. [Evaluation](#evaluation)
8. [Dashboard](#dashboard)
9. [Configuration](#configuration)
10. [How it works](#how-it-works)
11. [Development](#development)
12. [Troubleshooting](#troubleshooting)
13. [Known limitations](#known-limitations)

## Requirements

| Need | Version | Used for |
| --- | --- | --- |
| macOS or Linux | | Windows is untested |
| Conda (Miniforge or Miniconda) | any recent | the Python environment |
| Tesseract OCR | 5.x | reading depth labels, legend labels and track headers |
| Node.js | 20 or newer | building the dashboard front end (dashboard only) |
| Memory | 8 GB | the largest scans peak at about 3.7 GB; see `runtime.memory_budget_mb` |
| Internet | once | creating the environment and a one-time 45 MB download of the ResNet weights |

After the first run everything works offline. The dashboard's AI features are optional and need an API key.

## Installation

```bash
# 1. OCR engine
brew install tesseract                 # macOS
# sudo apt-get install tesseract-ocr   # Debian / Ubuntu

# 2. Python environment (Python 3.12, named "lithology"), installs the `lith` command
conda env create -f environment.yml
conda activate lithology

# 3. Check the installation
make check                             # lint + tests; dataset tests skip until the data is in place
```

`environment.lock.yml` records the exact versions the results were produced with. PyTorch comes from conda-forge
on purpose: it then shares one OpenMP runtime with NumPy and scikit-learn (a pip build of PyTorch loads a second
one, which can crash or hang).

## Placing the data

The data is **not** in this repository. It belongs to the challenge organisers and is several gigabytes, so it
stays local: `.gitignore` keeps only empty placeholder folders. Put it at the repository root exactly like this:

```
data-underneath/
├── dataset/
│   ├── images/              # development scans: .tif / .tiff (also .png, .jpg)        required
│   ├── bbox_annotation/     # one box file per scan: <scan name>.json                     recommended
│   ├── bbox_vis/            # organiser preview pictures, not read by the code             optional
│   └── test/
│       ├── images/          # evaluation scans
│       └── annotations/     # box files for the evaluation scans
└── example/                 # the organisers' worked example                               optional
    ├── <name>.tif
    ├── <name>.json
    └── example.csv
```

Why this layout:

- **Fixed paths for the tools that need them.** `lith run` accepts any folder, but the Makefile targets,
  `lith check-data`, the labelling tools and the dashboard read `dataset/images` and `dataset/bbox_annotation`.
  Keeping the data there means every command works without extra arguments.
- **Box files are matched by name.** A scan `images/Well_A.tif` uses the box file `Well_A.json`. It is looked
  for in a sibling folder called `bbox_annotation/` or `annotations/`, then beside the scan. Both organiser
  layouts (development and test) therefore work as delivered.
- **Box files are optional.** Without one, the built-in layout detector finds the three regions itself. It is
  slower (about 5 s more per scan) and slightly less accurate, so use the box files when you have them.
- **`example/` turns on the golden test.** When present, `make check` also runs the full pipeline on the worked
  example and fails if accuracy on it drops.
- **Duplicates and stray files are skipped.** Byte-identical copies of a scan and non-image files such as
  `Thumbs.db` are ignored and listed in the run log.

A box file holds three pixel boxes, `[x0, y0, x1, y1]` with the origin at top left and `x1`, `y1` exclusive:

```json
{
  "image_name": "Well_A.tif",
  "width": 2550,
  "height": 13200,
  "bounding_boxes": {
    "depth_column": [463, 6473, 541, 10828],
    "interpreted_lithology_column": [536, 6473, 719, 10828],
    "lithology_legend": [21, 3564, 2419, 4016]
  }
}
```

To check what the pipeline will see, run `lith check-data`. It warns about scans without a box file and box files
whose size does not match their scan, reports skipped files, and writes `data/manifest.json`.

## Running the pipeline

```bash
conda activate lithology

lith run dataset/images --out out                    # development scans
lith run dataset/test/images --out out_test          # evaluation scans
lith run path/to/Well_A.tif --out out_one            # a single scan
```

| Option | Effect |
| --- | --- |
| `--out DIR` | output folder (created if missing) |
| `--no-images` | skip the annotated PNGs (faster) |
| `--workers N` | scans processed in parallel (default 3, still limited by the memory budget) |
| `--detect-boxes` | ignore box files and detect the three regions (for comparison) |
| `--config FILE` | override settings, see [Configuration](#configuration) |

Each scan runs in its own process. A scan that fails or runs out of memory is reported in `run_summary.csv` and
never stops the batch. Typical cost on a laptop CPU: 8 to 10 s per scan, about 4 minutes for 51 scans.

## Outputs

Per scan, in the `--out` folder:

| File | Content |
| --- | --- |
| `<scan>.csv` | the submission file: `top_depth, bottom_depth, lithology, top_y, bottom_y, cluster_id` |
| `<scan>.png` | the lithology column with intervals, boundary depths, names and a colour key |
| `diagnostics/<scan>.csv` | the same rows plus `confidence`, `notes`, `status` and `alternatives` |
| `diagnostics/<scan>_legend.json` | legend entries read: label, swatch box, fill colour, rock or symbol |
| `diagnostics/<scan>_naming.json` | per group of beds: every opinion, its margin and the decision |
| `diagnostics/<scan>_layout.json` | where the three regions came from (box file or detector) and the detector's confidence |

For the whole run: `run_summary.csv` (time, peak memory and warnings per scan) and `run_meta.json` (git commit and
configuration hash, so every result can be reproduced).

`status` tells you how far to trust a name: `named` (the evidence agrees), `similar` (narrowed to two or three
look-alike legend entries, listed in `alternatives`) or `unresolved` (no entry matches clearly). The CSV always
holds the best guess; the diagnostics say how certain it is.

## Finding the three regions

Every scan needs three regions: the depth column, the interpreted lithology column and the legend.

1. **Box file present:** its boxes are used.
2. **No box file:** the layout detector finds them from the page itself (ruled lines, OCR of the depth labels and
   track headers, the legend's rock words). No training and no stored templates are involved.

To measure the detector against the box files:

```bash
lith layout dataset/images --out out/layout          # overlap (IoU) per region, speed, preview pictures
lith run dataset/images --out out_detect --detect-boxes   # the full pipeline on detected regions
```

On the 48 development scans the detected regions overlap the organisers' boxes (mean IoU) by 0.934 (lithology
column), 0.945 (legend) and 0.856 (depth column).

## Evaluation

Hand labels live in `labels/`, one `<scan>.csv` per labelled scan in the same format as the submission CSV.
`labels/manifest.yaml` marks each one `dev` (used while improving the pipeline) or `holdout` (scored once at the
end, from well families never inspected during development).

```bash
lith eval --pred out --labels labels --split dev     # depth-weighted name accuracy, boundary F1 (within 2 ft),
                                                     # depth error, grouping (adjusted Rand index)
lith audit --pred out                                # naming against spot labels read by eye (labels/audit)
```

Both append a row to `eval/results_log.csv` or `eval/audit_log.csv` with the git commit, configuration hash and
dataset hash, so any two runs can be compared. `lith labels-make` and `lith labels-import` build and read back a
labelling kit (pictures plus pre-filled CSVs to correct by hand).

## Dashboard

A local web app over the pipeline's results.

| Page | What it shows |
| --- | --- |
| CSV | the submission CSV row by row, run time and memory, CSV download |
| Image | the lithology column with every bed labelled at its depth, or the original scan with the three regions highlighted (box file or detector, switchable) |
| Layout | the detector's scores per region, detected against annotated boxes, and detection on an uploaded scan |
| Insights | all logs as one dataset: rock mix per well, wells correlated on one depth axis, and questions answered with the SQL shown |

**Run a scan** (top right) processes a new scan in place: drop a `.tif` or `.png`, with or without its `.json`.

```bash
lith run dataset/images --out out    # 1. results to show (once)
make dash-data                       # 2. database and scan tiles in dashboard/.cache (holdout scans hidden)
make dash                            # 3. build the front end, serve http://127.0.0.1:8000 and open it
```

Re-run steps 2 and 3 after a new pipeline run, and restart `make dash` after updating the code. `make dash-demo`
also shows the holdout scans. The dashboard shows the batch results in `out/` for scans found in
`dataset/images`; other scans go through **Run a scan**.

**AI (optional).** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` and
`OPENAI_MODEL` (plus `OPENAI_BASE_URL` for any OpenAI-compatible endpoint). Without a key, the charts and computed
summaries still work, and earlier AI answers are served from the local cache. The AI never changes the pipeline's
output; it only answers questions, through read-only SQL on an isolated copy of the data.

## Configuration

Every threshold lives in `config/default.yaml`, in physical units (feet, millimetres, inches) or as a fraction of a
box, never in raw pixels, so one setting works across scan resolutions. To change settings, write a small YAML
file with only the keys you want to change and pass it with `--config`:

```yaml
# my.yaml
runtime:
  workers: 2
  memory_budget_mb: 4000     # for a machine with less memory
ai:
  tier: 1                    # classical opinions only (no PyTorch needed, about 4 s faster per scan)
```

The configuration hash is recorded with every result.

## How it works

| Step | Module | Method |
| --- | --- | --- |
| 1. Regions | `layout.py` | box file, or the layout detector (rules, OCR of labels and headers, legend vocabulary) |
| 2. Depth scale | `scale.py` | Tesseract on windows of the depth column in three orientations, RANSAC line fit with a second reading per number (tick marks read as an extra digit), checked against the file-name scale times the scan resolution |
| 3. Beds | `segment.py`, `features.py` | colour and texture descriptors per block, change-point detection (`ruptures` KernelCPD), a second colour-only pass for thin beds, edges refined to the pixel row |
| 4. Groups | `cluster.py` | agglomerative clustering with a distance threshold, so the number of rocks is never fixed |
| 5. Legend | `legend.py` | swatches found by ink or colour and repaired from the legend's own grid; labels by OCR with spelling correction; rock or symbol decided by the words, else by the swatch picture |
| 6. Names | `naming.py` | independent opinions vote: two-way pixel template, stroke-skeleton distance, stroke shapes, and the frozen ResNet-18 only for undecided groups |
| 7. Checks | `checks.py` | ordering, depth increasing, minimum thickness; a failed check lowers confidence |
| 8. Output | `render.py` | the CSV and the annotated image |

## Development

```bash
make check          # ruff lint and format check, then pytest
make lock           # record exact package versions after changing environment.yml
```

Unit tests run on synthetic logs drawn with exact ground truth, so they need no data. Continuous integration runs
the same checks on every push.

```
src/lithology/      pipeline (one module per step), command line, evaluation
src/lithology/dash/ dashboard back end (FastAPI, DuckDB)
dashboard/          dashboard front end (React, Vite)
config/             default settings
labels/             hand labels, spot labels, labelling-kit selection
eval/               results logs
tests/              unit and end-to-end tests
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `TesseractNotFoundError` | Tesseract is not installed or not on `PATH`; install it (see [Installation](#installation)) |
| a scan's depths are empty in the CSV | fewer than four depth labels could be read; check the depth-column box and `run_summary.csv` warnings |
| `_layout.json` says `detected` although a box file exists | its name or folder does not match the scan; see [Placing the data](#placing-the-data) |
| a worker runs out of memory | lower `runtime.memory_budget_mb` or `runtime.workers` |
| the first run pauses for a while | the one-time ResNet weight download (45 MB) into `data/cache/torch` |
| the dashboard says an endpoint "did not return data" | the server is older than the page: stop it and run `make dash` again |
| Run a scan stays on one step for minutes | restart the dashboard; if it persists, run the same scan with `lith run` to see the error |

## Known limitations

- **Black-and-white logs.** Patterns in the column are drawn denser than in the legend, so most black-and-white
  beds are named with a flag rather than with confidence.
- **Thin beds on low-resolution logs.** Beds thinner than about 3 ft on 1 in / 100 ft scans blend with their
  neighbours and can get a mixed name.
- **Two-letter legend labels** (such as "Ss") are often missed by OCR, and beds of that rock take the nearest
  other entry.
- **Look-alike swatches** that differ only in sparse marks on white can be confused; such beds are flagged.
