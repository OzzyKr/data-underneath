# data-underneath

Automated lithology extraction from scanned mud-log images (Viridien HackTheBurgh, *Decoding the Subsurface*).

For each log the pipeline finds the lithology intervals, groups beds drawn with the same pattern, names them from
the log's own legend, converts pixel rows to depth, and writes a CSV plus an annotated image. It runs offline on a
laptop CPU and trains nothing: classical image processing and OCR, with a frozen ResNet-18 as a second opinion only
where the classical methods disagree.

## Installation

Requires macOS or Linux, conda, Tesseract 5, about 8 GB of memory, and Node.js 20+ for the dashboard.

```bash
brew install tesseract                 # Linux: sudo apt-get install tesseract-ocr
conda env create -f environment.yml    # Python 3.12 environment "lithology", installs the `lith` command
conda activate lithology
make check                             # lint and tests
```

The first run downloads the ResNet weights (45 MB); after that everything works offline.

## Placing the data

The data belongs to the organisers and is several gigabytes, so it is not in git (`.gitignore` keeps only empty
placeholder folders). Place it at the repository root:

```
dataset/
├── images/              development scans (.tif, .png, .jpg)
├── bbox_annotation/     one box file per scan: <scan name>.json
└── test/
    ├── images/          evaluation scans
    └── annotations/     their box files
example/                 optional: the organisers' example; enables the golden accuracy test
```

- The Makefile, `lith check-data` and the dashboard read `dataset/images` and `dataset/bbox_annotation`, so
  keeping the data there means every command works without extra arguments.
- A box file is matched to its scan by name (`Well_A.tif` uses `Well_A.json`), in a sibling `bbox_annotation/` or
  `annotations/` folder or beside the scan.
- Without a box file, the built-in layout detector finds the three regions (depth column, lithology column, legend)
  itself: about 5 s slower per scan and slightly less accurate.

Run `lith check-data` to see which scans and box files will be picked up.

## Running

```bash
lith run dataset/images --out out              # development scans
lith run dataset/test/images --out out_test    # evaluation scans
```

Options: `--no-images` (skip annotated PNGs), `--workers N`, `--detect-boxes` (ignore box files),
`--config FILE`. Expect 8 to 10 s per scan.

Each scan produces `<scan>.csv` (submission format: `top_depth, bottom_depth, lithology, top_y, bottom_y,
cluster_id`) and `<scan>.png`. `diagnostics/` adds a confidence and status per bed (`named`, `similar` or
`unresolved`), and `run_summary.csv` lists time, memory and warnings per scan.

To score against hand labels in `labels/`: `lith eval --pred out --labels labels`.

## Dashboard

```bash
make dash-data    # build the dashboard data from out/
make dash         # serve http://127.0.0.1:8000
```

Pages: CSV, Image (beds on the scan), Layout (detector scores) and Insights (all logs as one dataset). **Run a scan**
processes a new scan, with or without its box file. AI answers on the Insights page are optional: copy
`.env.example` to `.env` and add an API key.

## Configuration

All thresholds live in `config/default.yaml`, in physical units rather than pixels. Override any of them with a
small YAML file passed to `--config`, for example `runtime: {memory_budget_mb: 4000}` on a machine with less memory.
