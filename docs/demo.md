# Demo script (5 minutes)

Every number below can be reproduced with the command next to it, and was measured at commit `b56973b`
(`out/run_meta.json`). If a number changes, change it here before presenting it.

## 0. Before the demo (once)

```bash
lith fetch-weights                       # caches the tier-2 weights; everything after this runs offline
lith run dataset/images --out out        # 51 logs, 0 errors, ~10 s/log, peak ~2.3 GB
lith dash-build && lith dash --open      # dashboard at http://127.0.0.1:8000
```

## 1. The problem (30 s)

Legacy mud logs are scanned images. The lithology column holds the geology, but nothing can query it.
For each log we produce interval CSVs (top/bottom pixel, cluster, lithology, top/bottom depth) and an annotated image.
Everything runs locally: classical image processing plus Tesseract OCR, no trained model, no cloud.

## 2. Live run on a log (60 s)

```bash
lith run "dataset/images/Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft.tif" --out /tmp/demo
```

- Open the CSV and `/tmp/demo/*.png`. Point out the depth labels at each boundary and the cluster ids, and that the
  names come from the log's own legend.
- **Boxes:** the run uses the supplied annotation JSON. Without a JSON (or with `--detect-boxes`), the layout detector
  finds the boxes itself.

## 3. How we know it is right (60 s)

```bash
lith eval --pred out --labels labels     # hand-labelled log: accuracy 0.835, boundary F1 0.848, depth MAE 0.12 ft
lith audit --pred out                    # 98 eye-audit spot labels on 15 logs: 88 right, 22 flagged, 4 confident-wrong
```

Be upfront that ground truth is thin:
- Only one log is fully hand-labelled, and it was also used for tuning.
- The 98 audit labels were read by eye; they were not checked by a geologist.

## 4. Bonus: finding the boxes without the JSON (60 s)

Show the dashboard → **Layout**.
- Scores are IoU with the annotated boxes.

  | Box | Dev (48 pages, 12 vendors) | Holdout (3 pages, never tuned on) |
  |---|---|---|
  | Depth column | 0.856 | 0.83 |
  | Lithology column | 0.934 | 0.65, right on 2 of 3 |
  | Legend | 0.945 | 0.98 |

- Click the worst page to show what it gets wrong, and point out the confidence flag. On dev and holdout, every box
  with confidence ≥ 0.5 was right; low-confidence boxes are flagged for review.
- **Detect on an image:** upload a log and get its boxes in about 5 s.

## 5. Robustness (30 s)

Stress test, one page per vendor (lithology-column IoU):
- Clean: 0.92
- Noise: 0.89
- Greyscale: 0.84
- 70% resolution: 0.78
- 0.1 in scanner skew: about 0.58, **the known weak spot**

## 6. Where this goes (60 s)

Bypassed-pay screening (branch `bypassed-pay`, Phase A):
- The 42 offshore logs match BOEM borehole records on API.
- Public completion reports give perforated depths for every producing completion.
- Next steps: digitise the gas curves and compare the shows on our logs with the sands that were actually produced.

This is screening, not investment advice.

## Limitations to say before anyone asks

- Black-and-white hatch patterns are the hardest to name; doubtful groups are flagged rather than guessed.
- Accuracy is measured on thin ground truth (see 3).
- Scanner skew hurts box detection when there is no JSON.
- The Insights page's AI answers are optional. They use a cloud model only if API keys are configured; the pipeline
  itself never does.
