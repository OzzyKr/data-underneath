# How to label (about 10-15 minutes per log)

You are correcting the pipeline's draft. For each log `NN_name` there are four files:

| File | What it is |
| --- | --- |
| `NN_name.png` | The log window. Red ticks = draft boundaries. Red `#n NAME` = draft interval n and its guessed rock. Blue ruler (right) = depth in ft, a mark every 5 ft, a number every 10 ft |
| `NN_name_legend.png` | The log's own legend: what each rock pattern looks like and its name |
| `NN_name_legend_names.txt` | Names the pipeline read from that legend (may be incomplete or wrong; the picture is the truth) |
| `NN_name.csv` | **The only file you edit.** One row per interval: `n, top_depth, bottom_depth, lithology, note` |

## Steps for one log

1. Open `NN_name.png` and `NN_name.csv` side by side (VS Code: drag the CSV to the right half).
2. Look at `NN_name_legend.png` once, so you know which pattern is which rock.
3. Walk down the CSV row by row, comparing with the picture:
   - **Name wrong?** Type the correct name exactly as the legend writes it (copy from the legend picture).
   - **Boundary in the wrong place?** Read the depth where the pattern really changes off the blue ruler and fix
     `top_depth` / `bottom_depth`. Nearest foot is fine.
   - **Missing boundary** (pattern changes, no red tick)? Copy the row, then split the depths between the two rows.
   - **Extra boundary** (red tick, same pattern on both sides)? Delete one row and stretch the other over both.
   - **Can't tell?** Make your best guess and write a few words in `note`.
4. Rules: each row's `top_depth` must equal the previous row's `bottom_depth`. Keep the header line. Save.

Label the **dev** logs (01-04) first. The **holdout** logs (05-07) are scored only once at the end, so label
them carefully and don't tweak them afterwards. When done, say so: `lith labels-import` turns your CSVs into
labels and reports any rule breaks.
