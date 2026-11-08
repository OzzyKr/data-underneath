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

lith fetch-weights                     # once, online: cache the tier-2 model weights (runs are offline after this)
lith check-data                        # audit dataset/, write data/manifest.json
lith run dataset/images --out out      # CSV + annotated PNG per log, plus run_summary.csv
                                       #   boxes: the annotation JSON when there is one, detected when there is not
                                       #   --detect-boxes: always detect (bonus); --tier 1: classical naming only
lith eval --pred out --labels labels   # score against hand labels (dev split), append to eval/results_log.csv
lith audit --pred out                  # score naming against eye-audit spot labels (labels/audit)
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
| `diagnostics/<doc>.csv` | Same rows plus `confidence`, `notes`, `status` (`named` / `similar` = narrowed to 2-3 look-alike legend entries / `unresolved`) and `alternatives` |
| `<doc>.png` | Annotated lithology column: intervals, boundary depths, labels, colour key |
| `run_summary.csv`, `run_meta.json` | Per-document runtime, peak memory, warnings; commit and config hash |
| `diagnostics/<doc>_naming.json`, `_legend.json` | Naming opinions per rock group; legend entries with swatch colours |

## Dashboard (Stage 2)

A local web app over the pipeline's outputs: **CSV** (the submission CSV, row by row), **Image** (the scan with
the predicted beds drawn on it, linked to the CSV) and **Insights** (all logs as one dataset, with an AI that
answers questions by writing read-only SQL). A scan can also be dropped in and run live.

```bash
make run          # pipeline -> out/ (once)
make dash-data    # out/ -> dashboard/.cache (DuckDB + scan tiles; holdout logs hidden)
make dash         # build the front end, serve http://127.0.0.1:8000 and open it
# final demo only: make dash-demo   (includes the holdout logs)
```

AI: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (Claude, default `claude-opus-5-5`), or
`OPENAI_API_KEY` + `OPENAI_MODEL` (+ `OPENAI_BASE_URL` for any OpenAI-compatible endpoint). Without a key the
Insights page works offline: charts and computed summaries always, AI notes and answers from the cache of earlier
runs. Front-end development: `lith dash --dev` (API with reload) plus `cd dashboard && npm run dev` (port 5173).

## How it works

| Step | Module | Method |
| --- | --- | --- |
| 1. Load | `io.py` | Lazy TIFF open, strip-wise crop of only the needed regions |
| 2. Depth scale | `scale.py` | Tesseract on windows of the depth column, all 3 text orientations; exhaustive RANSAC line fit with "dropped digit" hypotheses; cross-checked against filename scale x TIFF DPI |
| 3. Segment | `segment.py`, `features.py` | Per-block colour + pattern descriptors, change points (`ruptures` KernelCPD), boundary refinement to the pixel row, blank-paper detection |
| 4. Cluster | `cluster.py` | Agglomerative clustering with a distance threshold (no fixed k) |
| 5. Legend | `legend.py` | Swatch detection (ink or colour), repaired from the legend's own grid (size, columns, label offset); OCR labels (right-of or below layouts) with spelling fixes; rock vs symbol from label text, else from the swatch picture |
| 6. Naming | `naming.py` | Opinions: two-way pixel template, classical descriptor, pretrained CNN (tier 2), stroke shapes; decisive opinion names the bed, otherwise a flagged best guess (rank vote + look-alike rules) |
| 7. Checks | `checks.py` | Contiguity, containment, depth increasing, minimum thickness; failures lower confidence |
| 8. Output | `render.py` | CSV + annotated image (drawing code ported from the organisers' notebook) |

Every threshold lives in `config/default.yaml`, in feet or as a fraction of a box, never raw pixels.

## Engineering rules

- Same input + same config = same output. No network calls in the default run.
- Every change is scored; results are logged with git commit, config hash and dataset hash.
- Accuracy is claimed only on a frozen holdout (see `labels/manifest.yaml`), split by document family.
- A failed step degrades and is reported; a batch never stops on one bad document, even when its worker process
  dies (e.g. out of memory): the documents lost with it are retried one at a time.
- Each document runs in its own process, so runtime and peak memory (including the annotated image) are measured
  per document. Documents start largest first and only while their estimated peak memory (read from the image
  header) fits `runtime.memory_budget_mb` (6000 MB, for an 8 GB machine).
