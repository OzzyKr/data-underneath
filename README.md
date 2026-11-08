# data-underneath

Lithology interval extraction from scanned mud logs (Viridien HackTheBurgh, *Decoding the Subsurface*).

Per log: interval segmentation, pattern clustering, legend-based naming and pixel-to-depth conversion, written as
the submission CSV plus an annotated image. CPU-only, offline and training-free: classical computer vision and
Tesseract OCR, with a frozen ImageNet ResNet-18 consulted only for groups the classical matchers leave undecided.

## WINDOWS (untested)

This `windows` branch adds Windows support: peak memory without the Unix-only `resource` module, UTF-8 for every
text file, and Tesseract lookup. It has not been validated on Windows hardware; CI runs the test suite on
`windows-latest`.

- Tesseract: install the UB Mannheim build; `C:\Program Files\Tesseract-OCR` is found automatically, otherwise set
  `TESSERACT_CMD` to the full path of `tesseract.exe`.
- Create the environment from a Miniforge or Anaconda prompt: `conda env create -f environment.yml`.
- No `make` on Windows; run the targets directly: `ruff check . && pytest -q` (check), `lith dash-build` (dash-data),
  `cd dashboard && npm install && npm run build && cd ..` then `lith dash --open` (dash).
- Enable Windows long-path support if the repository sits in a deep folder.

## Installation

macOS or Linux, conda, Tesseract 5, ~8 GB RAM; Node.js 20+ for the dashboard.

```bash
brew install tesseract                 # Linux: apt-get install tesseract-ocr
conda env create -f environment.yml    # Python 3.12 env "lithology"; installs the `lith` CLI
conda activate lithology
make check                             # ruff + pytest
```

The ResNet weights (45 MB) are fetched on first use; subsequent runs need no network.

## Data layout

The dataset is organiser-owned and several GB, so it is not versioned; `.gitignore` retains placeholder folders only.

```
dataset/
├── images/              development scans (.tif, .png, .jpg)
├── bbox_annotation/     box files, <scan stem>.json
└── test/
    ├── images/          evaluation scans
    └── annotations/     box files
example/                 optional: organisers' worked example; enables the golden regression test
```

- `dataset/images` and `dataset/bbox_annotation` are the default paths for the Makefile targets, `lith check-data`
  and the dashboard; `lith run` accepts any directory.
- Box files resolve by scan stem from a sibling `bbox_annotation/` or `annotations/` directory, then from the
  scan's own directory.
- Scans without a box file fall back to the layout detector for the depth column, lithology column and legend
  (~5 s extra per scan, slightly lower region accuracy).

`lith check-data` reports scan/box-file pairing, size mismatches and skipped files, and writes `data/manifest.json`.

## Usage

```bash
lith run dataset/images --out out              # development set
lith run dataset/test/images --out out_test    # evaluation set
```

Flags: `--no-images` (skip annotated PNGs), `--workers N`, `--detect-boxes` (ignore box files), `--config FILE`.
Throughput is 8-10 s per scan.

Outputs per scan: `<scan>.csv` in submission format (`top_depth, bottom_depth, lithology, top_y, bottom_y,
cluster_id`) and `<scan>.png`. `diagnostics/` adds per-bed confidence and status (`named` | `similar` |
`unresolved`); `run_summary.csv` records runtime, peak memory and warnings per scan.

Scoring against the hand labels in `labels/`: `lith eval --pred out --labels labels`.

## Dashboard

```bash
make dash-data    # DuckDB + scan tiles from out/
make dash         # serves http://127.0.0.1:8000
```

Pages: CSV, Image (beds on the scan), Layout (detector scores) and Insights (cross-log analysis). **Run a scan**
executes the pipeline on an uploaded scan; the box file is optional. LLM answers on Insights are optional and need
a provider key in `.env` (template: `.env.example`).

## Configuration

Thresholds are defined in physical units in `config/default.yaml`. `--config` merges a partial YAML override, e.g.
`runtime: {memory_budget_mb: 4000}` for machines with less memory.
