# Handoff — lithology extraction (Viridien HackTheBurgh) — 2026-11-07

Branch `ozan` (pushed, CI green, 68 tests). Run: `conda activate lithology && lith run dataset/images --out out`.
Plan: `~/.claude/plans/keep-in-mind-this-quirky-codd.md` · Docs: [plan](https://claude.ai/artifact/4P8kJqch2Af1dR5ZLP7Zb2), [recap](https://claude.ai/artifact/2Hu2jgYnup6Th33Ze9MsjC)

## Decisions
- Engineering-first: classical CV + OCR, deterministic, laptop CPU, every threshold in `config/default.yaml` (ft, not px).
- AI only where classical can't decide: tier 2 = frozen ResNet-18 second opinion, asked only for open groups.
- Naming = 3 opinions (pixel template · classical · pretrained). No agreement → `Unresolved` + "computer vision could not differentiate: A or B?".
- Holdout logs (05–07) come from families never inspected; scored once at the freeze.
- Viridien: ranked vs other teams (no fixed metric); test images include boxes; detecting boxes ourselves = extra points.

## Trade-offs
| Choice | Gain | Cost |
| --- | --- | --- |
| No trained model | No overfitting, explainable | Weak on black-and-white pattern naming |
| Abstain (`naming.abstain: true`) | ~0 confident errors | Unresolved always scores wrong (`false` = flagged best guess) |
| Tier 2 model on | Better b&w names | 5.7 → 9.5 s/log, +~330 MB |
| Thin beds ≥0.5 ft (colour only, A-b-A rule, ≥6 px) | Finds 1 ft interbeds | B&W logs only resolve ≥2 ft |
| Whole-image decode | Simple, all 51 files work | Peak 3.1 GB on the 313,800 px log |

## Standing (measured)
| Measure | Result |
| --- | --- |
| Logs processed | 51/51, 0 errors |
| Depth scale | Plausible on 51/51; example MAE 0.12 ft |
| Boundaries (example, ±2 ft) | F1 0.91 |
| Grouping (example) | ARI 0.83 |
| Names (example, b&w) | 33% right (lenient), 62% Unresolved, ~6% wrong |
| Names (MC765 colour) | All groups right via pixel match |
| All logs | 75% of groups named, 24% Unresolved |
| Speed / memory | 9.5 s/log mean, 29.5 s max; ~1 GB median, 3.1 GB max |

Only interval ground truth: `example/example.csv` (log06). Scores beyond it wait on hand labels.

## Labelling (in progress)
Kit in `labels/kit/` (see `HOW_TO_LABEL.md`). Dev: 01 mc765 (edited, **uncommitted**), 02 rio, 03 shenzi, 04 mc761, 08 bw_log04. Holdout: 05 shell, 06 xom, 07 neptune.
Conventions: write the true rock (never "Unresolved"); b&w sand = `Sandstone`; label thin beds too.
When done: `lith labels-import` then `lith eval --pred out --labels labels --split dev`.

## Known weaknesses
- B&W naming: column strokes differ from legend; Shale vs Clay/Silty Sandstone near-identical.
- Thin beds of look-alike patterns (shenzi cyan: Limestone/Dolomitic/Chalk) → Unresolved.
- Legend: drilling symbols filtered; junk fragments rare but possible (FREEMAN 64/~100 entries).
- Naming thresholds set on few logs: validate on labels.

## Next
1. Import labels, score dev; tune abstain on/off, thin-bed min 0.5 vs 1 ft, naming thresholds.
2. Bonus: detect the 3 boxes (use `lithology_track_label` header text in 24 JSONs; score IoU vs 51 boxes).
3. Cut peak memory; freeze (score holdout once, tag `v1.0`); run test images unchanged.
4. Stage 2 dashboard is built (see "Dashboard" below): add an AI key to `.env`, rehearse, then `make dash-demo`.

## Dashboard (Stage 2)
- `make dash-data` then `make dash` -> http://127.0.0.1:8000. Pages: CSV, Image (Column = lithology column + depth ruler + labelled beds, zoom lands on the selected bed;
  Original scan = the uncropped page, true to scale, spotlighting the three regions read, beds labelled), Insights
  (fixed-SQL charts, AI notes, Ask the data with visible SQL), Run a scan (live pipeline run, ~9 s for log06).
- Back end `src/lithology/dash/` (store/tiles/insights/ai/jobs/app), front end `dashboard/` (Vite + React + Tailwind).
- Holdout logs are hidden unless `lith dash-build --all` (`make dash-demo`) - only for the final demo.
- AI provider switch: `.env` (`.env.example`); offline mode serves cached notes/answers - ask the demo questions
  once with a key before the demo so they work offline too.
- Rock colours: the legend's hue snapped to a colour-blind-validated palette; Unresolved is hatched grey.
