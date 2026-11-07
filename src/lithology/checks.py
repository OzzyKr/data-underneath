"""Physical sanity checks on a document's intervals. Failures lower confidence and are reported, never hidden."""

from __future__ import annotations

import math

from lithology.config import ChecksConfig
from lithology.models import Box, Interval


def check_intervals(intervals: list[Interval], column: Box, cfg: ChecksConfig) -> list[str]:
    """Validate ordering, containment, depth monotonicity and thickness. Mutates confidence/notes in place."""
    problems: list[str] = []
    for prev, cur in zip(intervals, intervals[1:], strict=False):
        if cur.top_y < prev.bottom_y:
            problems.append(f"overlap at y={cur.top_y}")
            cur.confidence *= 0.5
            cur.notes.append("overlaps previous interval")
    for iv in intervals:
        if iv.top_y < column.y0 or iv.bottom_y > column.y1:
            problems.append(f"interval {iv.top_y}-{iv.bottom_y} leaves the lithology column")
            iv.confidence *= 0.5
            iv.notes.append("outside column")
        if not (math.isfinite(iv.top_depth) and math.isfinite(iv.bottom_depth)):
            iv.notes.append("no depth scale")
            iv.confidence *= 0.5
            continue
        if cfg.depth_monotonic and iv.bottom_depth <= iv.top_depth:
            problems.append(f"depth does not increase at y={iv.top_y}")
            iv.confidence *= 0.3
            iv.notes.append("depth not increasing")
        if iv.bottom_depth - iv.top_depth < cfg.min_thickness_ft:
            iv.confidence *= 0.7
            iv.notes.append(f"thinner than {cfg.min_thickness_ft:g} ft")
    return problems
