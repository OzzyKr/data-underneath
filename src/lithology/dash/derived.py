"""Well-level tables derived from the per-log intervals, for questions about wells and salt.

The raw ``intervals`` table is one row per bed per log, and one well can have several logs (scales, MD/TVD versions,
sidetracks). Questions such as "where does the salt end?" are easy to get wrong on raw beds: a 4 ft salt stringer
far below the salt body becomes "the base", a well is counted once per log, and a log that starts inside the salt
makes its first row look like the top of salt. These tables settle those choices once, in code:

wells  one row per well:  names, logs, depth range.
salt   one row per log that crosses a salt body (``main_log`` marks one log per well): the main body (top, base,
       thickness), whether its top is really seen on the log, the mixed salt/sediment zone just below the base,
       and the sand below the salt.
"""

from __future__ import annotations

import pandas as pd

MERGE_GAP_FT = 50.0  # non-salt gaps up to this thick are inclusions inside one salt body
MIXED_ZONE_FT = 500.0  # salt beds within this distance below the base form the "mixed zone", not the body
MIN_BODY_FT = 100.0  # thinner salt is a stringer, not a body
TOP_SEEN_WINDOW_FT = 100.0  # the top of salt is "seen" only if the log shows other rock just above it:
TOP_SEEN_MIN_FT = 15.0  # at least this much non-salt rock within TOP_SEEN_WINDOW_FT above the body top


def wells_table(logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, g in logs.groupby("well_group", dropna=False):
        names = sorted({str(t) for t in g["title"].dropna()})
        rows.append(
            {
                "well": well,
                "other_names": " | ".join(n for n in names if n != well),
                "n_logs": len(g),
                "log_titles": " | ".join(names),
                "depth_refs": ",".join(sorted({str(d) for d in g["depth_ref"].dropna()})) or None,
                "top_ft": g["top_ft"].min(),
                "bottom_ft": g["bottom_ft"].max(),
                "kind": ",".join(sorted({str(k) for k in g["kind"].dropna()})),
            }
        )
    return pd.DataFrame(rows)


def _bodies(salt: pd.DataFrame) -> list[list[float]]:
    """Salt beds of one log merged into bodies, bridging non-salt gaps up to MERGE_GAP_FT."""
    out: list[list[float]] = []
    for t, b in zip(salt["top_depth"], salt["bottom_depth"], strict=True):
        if out and t - out[-1][1] <= MERGE_GAP_FT:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] += b - t
        else:
            out.append([t, b, b - t])
    return out


def salt_table(logs: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per log whose thickest salt body is at least MIN_BODY_FT (bodies are per log; see ``well``)."""
    cols = [
        "well", "log_title", "depth_ref", "salt_top_ft", "top_seen", "salt_base_ft", "body_thickness_ft",
        "salt_in_body_ft", "mixed_zone_base_ft", "mixed_zone_salt_ft", "sand_below_base_ft", "thickest_sand_below_ft",
        "log_top_ft", "log_bottom_ft", "n_salt_bodies", "main_log",
    ]  # fmt: skip
    iv = intervals.dropna(subset=["top_depth", "bottom_depth"])
    rows = []
    for _, lg in logs.iterrows():
        d = iv[iv["log_id"] == lg["log_id"]].sort_values("top_depth")
        salt = d[d["rock_family"] == "Salt"]
        if salt.empty:
            continue
        bodies = [b for b in _bodies(salt) if b[1] - b[0] >= MIN_BODY_FT]
        if not bodies:
            continue
        top, base, salt_ft = max(bodies, key=lambda b: b[1] - b[0])
        first = d["top_depth"].min()
        lo = top - TOP_SEEN_WINDOW_FT
        above = d[(d["bottom_depth"] > lo) & (d["top_depth"] < top) & (d["rock_family"] != "Salt")]
        above_ft = float((above["bottom_depth"].clip(upper=top) - above["top_depth"].clip(lower=lo)).sum())
        below = salt[(salt["top_depth"] >= base) & (salt["top_depth"] < base + MIXED_ZONE_FT)]
        sand = d[(d["rock_family"] == "Sand") & (d["top_depth"] >= base)]
        sand_t = sand["bottom_depth"] - sand["top_depth"]
        rows.append(
            {
                "well": lg["well_group"],
                "log_title": lg["title"],
                "depth_ref": lg["depth_ref"],
                "salt_top_ft": round(top, 1),
                "top_seen": above_ft >= TOP_SEEN_MIN_FT,
                "salt_base_ft": round(base, 1),
                "body_thickness_ft": round(base - top, 1),
                "salt_in_body_ft": round(salt_ft, 1),
                "mixed_zone_base_ft": round(float(below["bottom_depth"].max()), 1) if len(below) else None,
                "mixed_zone_salt_ft": round(float((below["bottom_depth"] - below["top_depth"]).sum()), 1),
                "sand_below_base_ft": round(float(sand_t.sum()), 1),
                "thickest_sand_below_ft": round(float(sand_t.max()), 1) if len(sand) else 0.0,
                "log_top_ft": round(first, 1),
                "log_bottom_ft": round(float(d["bottom_depth"].max()), 1),
                "n_salt_bodies": len(bodies),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    out["main_log"] = False
    if len(out):
        # one representative log per well: the one showing the thickest salt body (other logs = other scales/versions)
        best = out.sort_values(["well", "body_thickness_ft"], ascending=[True, False]).groupby("well").head(1).index
        out["main_log"] = out.index.isin(best)
    return out
