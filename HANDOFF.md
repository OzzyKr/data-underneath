# Handoff — lithology extraction (Viridien HackTheBurgh) — 2026-11-07 (evening)

Branch `ozan` (pushed). Run: `conda activate lithology && lith run dataset/images --out out`, then `lith audit --pred out`
and `lith eval --pred out --labels labels --split dev`. Plans: `~/.claude/plans/keep-in-mind-this-quirky-codd.md`
(build), `~/.claude/plans/return-to-your-context-tingly-spindle.md` (accuracy pass). Docs: [plan](https://claude.ai/artifact/4P8kJqch2Af1dR5ZLP7Zb2), [recap](https://claude.ai/artifact/2Hu2jgYnup6Th33Ze9MsjC)

## Decisions
- Engineering-first: classical CV + OCR, deterministic, laptop CPU, every threshold in `config/default.yaml` (ft, not px).
- AI only where classical can't decide: tier 2 = frozen ResNet-18 second opinion, asked only for open groups.
- Naming opinions: pixel template (two-way) · classical · pretrained · stroke shapes (port of a teammate's prototype).
- **CSV always gets a legend name** (`naming.abstain: false`, decided 2026-11-07): Viridien ranks on correctness and
  "Unresolved" always scores wrong. Honesty lives in the diagnostics: `confidence`, `notes`, and `status`:
  `named` · `similar` (vision narrowed the bed to 2-3 look-alike entries of the bed's colour, listed in `alternatives`)
  · `unresolved` (no entry matches clearly). The dashboard shows these as "Similar types" (light hatch) and
  "Unresolved" (grey hatch).
- Look-alike rules (explainable): every opinion leans one way → that entry; variants of one rock split
  ("Clay-shale" / "Sandy clay-shale") → plainest entry; pixel-identical swatches (b&w Clay = Shale) → commoner rock
  (`naming.common_rocks`); a guess must have the bed's fill colour when any swatch does.
- Rock filter generalises (no fixed list): rock word → rock; drilling/show/accessory word → not; unknown words →
  decided by the swatch picture (fill texture = rock, single glyph = symbol). Spelling: "Silstone" → "Siltstone".
- Holdout logs (05–07: Shell_AC0336, xom, neptune) never opened; scored once at the freeze.
- Viridien: ranked vs other teams (no fixed metric); test images include boxes; detecting boxes ourselves = extra points.

## Trade-offs
| Choice | Gain | Cost |
| --- | --- | --- |
| No trained model | No overfitting, explainable | Weak on black-and-white pattern naming |
| Best guess in CSV (abstain off) | Every bed can score | A flagged guess is sometimes wrong (shown as Similar types / Unresolved) |
| "All opinions lean one way" names the bed | 8 fewer flags in the audit | 2 more wrong names (mc822 grey clay-shales) |
| Variants → plainest entry | Right at base-rock level | Wrong when the truth is the variant ("Silty clay-shale") |
| Tier 2 model on | Better b&w names | ~+1-3 s/log, +~330 MB |
| Thin beds ≥0.5 ft (colour only, A-b-A rule, ≥6 px) | Finds 1 ft interbeds | B&W logs only resolve ≥2 ft |
| Whole-image decode | Simple, all 51 files work | Peak 3.7 GB on the 527 Mpx log (scheduler keeps it alone) |

## Standing (measured 2026-11-07, `eval/audit_log.csv`, `eval/results_log.csv`)
| Measure | Before (e2bebd2) | Now |
| --- | --- | --- |
| Eye audit: 91 bed samples on 13 colour logs (`labels/audit`, labelled by eye) | 52/91 right, 16 wrong-as-answer | 80/91 right, 3 wrong-as-answer |
| Example log06 (b&w), share of depth named right (exact / lenient) | 20% / 33% | 82% / 94% |
| Example boundaries (±2 ft) / grouping ARI / depth MAE | F1 0.91 / 0.83 / 0.12 ft | unchanged |
| "Unresolved" or non-rock names in CSVs | 72,000 ft / 263 ft | 0 / 0 |
| Flagged depth (similar + unresolved): colour / b&w logs | 10% / 92% | 14% / 94% |
| Logs processed | 51/51 | 51/51, 0 errors |
| Speed / peak memory | 7.4 s/log / 2.9 GB (under-reported) | 9.6 s/log / 3.7 GB (measured after image) |

Eye-audit labels are Claude's reading of column vs legend, not ground truth: use them to compare runs, never as an
accuracy claim. The only official interval truth is `example/example.csv` (log06).

## Labelling (in progress)
Kit in `labels/kit/` (see `HOW_TO_LABEL.md`). Dev: 01 mc765 (edited, **uncommitted**), 02 rio, 03 shenzi, 04 mc761, 08 bw_log04. Holdout: 05 shell, 06 xom, 07 neptune.
Conventions: write the true rock (never "Unresolved"); b&w sand = `Sandstone`; label thin beds too.
When done: `lith labels-import` then `lith eval --pred out --labels labels --split dev`.
Eye-audit spot labels: `labels/audit/<doc>.csv` (windows mid-bed); score with `lith audit --pred out`.

## Known weaknesses
- B&W logs are still ~94% flagged: patterns are drawn denser than the legend; best next lever = read the mudlogger's
  text beside the column ("Sh:", "Sltst:", "Sdst:") as an extra opinion (outside the 3 boxes, ~+3 s/log).
- Look-alike sparse patterns on white (Trail Mtn IGNEOUS "X X" vs CHERT "^ ^") and same-colour fills (Rio orange
  SILTSTONE vs ANHYDRITE) are still guessed wrong (flagged).
- Clustering can merge two variants into one group (mc822 plain + silty clay-shale).
- Low depth-scale confidence on 5 logs is OCR noise in the depth column, checked by eye: Trail Mtn, g16786, Hadrian
  within ~1 ft; log08 ±3 ft (5 labels); ocs-g 16987 slope right, possible ~5 ft offset.
- FREEMAN (64→81 legend entries, interbedded 1-px colour stripes) is not auditable by eye.

## Next
1. Your eye check in the dashboard; decide whether "all opinions lean one way" should stay (see trade-offs).
2. B&W: mudlogger-text opinion (if wanted); bonus: detect the 3 boxes (IoU vs 51 boxes).
3. Freeze (score holdout once, tag `v1.0`); run test images unchanged.
4. Stage 2 dashboard: add an AI key to `.env`, rehearse, then `make dash-demo`.

## Dashboard (Stage 2)
- `make dash-data` then `make dash` -> http://127.0.0.1:8000. Pages: CSV, Image (Column = lithology column + depth ruler + labelled beds, zoom lands on the selected bed;
  Original scan = the uncropped page, true to scale, spotlighting the three regions read, beds labelled), Insights
  (fixed-SQL charts, AI notes, Ask the data with visible SQL), Run a scan (live pipeline run, ~9 s for log06).
- Back end `src/lithology/dash/` (store/tiles/insights/ai/jobs/app), front end `dashboard/` (Vite + React + Tailwind).
- Holdout logs are hidden unless `lith dash-build --all` (`make dash-demo`) - only for the final demo.
- AI provider switch: `.env` (`.env.example`); offline mode serves cached notes/answers - ask the demo questions
  once with a key before the demo so they work offline too.
- Rock colours: the legend's hue snapped to a colour-blind-validated palette; Unresolved is hatched grey; Similar types
  keep the guessed rock's colour with a fine hatch and show the look-alike entries ("A / B").
