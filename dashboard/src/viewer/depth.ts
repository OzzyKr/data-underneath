import type { Interval } from "../api";

/** Piecewise-linear pixel-row <-> depth mapping from the interval boundaries (as the pipeline's depth scale). */
export function depthMapper(intervals: Interval[]) {
  const pts = new Map<number, number>();
  for (const r of intervals) {
    if (r.top_depth != null && !pts.has(r.top_y)) pts.set(r.top_y, r.top_depth);
    if (r.bottom_depth != null && !pts.has(r.bottom_y)) pts.set(r.bottom_y, r.bottom_depth);
  }
  const ys = [...pts.keys()].sort((a, b) => a - b);
  const ds = ys.map((y) => pts.get(y)!);

  const interp = (xs: number[], vs: number[], x: number): number | null => {
    const n = xs.length;
    if (n < 2) return null;
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid;
      else hi = mid;
    }
    const span = xs[hi] - xs[lo];
    return span === 0 ? vs[lo] : vs[lo] + ((vs[hi] - vs[lo]) * (x - xs[lo])) / span;
  };
  const monotonic = ds.every((d, i) => i === 0 || d >= ds[i - 1]);
  return {
    depthAt: (y: number) => interp(ys, ds, y),
    yAt: (depth: number) => (monotonic ? interp(ds, ys, depth) : null),
  };
}

/** Round tick depths: at most ~maxTicks between lo and hi. */
export function depthTicks(lo: number, hi: number, maxTicks = 7): number[] {
  const steps = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000];
  const step = steps.find((s) => (hi - lo) / s <= maxTicks) ?? 20000;
  const out: number[] = [];
  for (let d = Math.ceil(lo / step) * step; d <= hi; d += step) out.push(d);
  return out;
}

export const shortDepth = (d: number) => (d >= 1000 ? `${(d / 1000).toFixed(d % 1000 ? 1 : 0)}k` : `${d}`);
