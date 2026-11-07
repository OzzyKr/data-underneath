import type { ReactNode } from "react";
import type { Interval, LogDetail } from "../api";
import { bedName, texture } from "../components/ui";
import { fmt } from "../format";

/** Shared drawing for the Column and Original scan views: depth ruler, bed labels and their leader lines. */

export const SELECT = "#8F74EA";

export function estimatePxPerFt(log: LogDetail, intervals: Interval[]): number {
  if (log.px_per_ft) return log.px_per_ft;
  const a = intervals.find((r) => r.top_depth != null);
  const b = [...intervals].reverse().find((r) => r.bottom_depth != null);
  return a && b && b.bottom_depth! > a.top_depth! ? (b.bottom_y - a.top_y) / (b.bottom_depth! - a.top_depth!) : 10;
}

type Mapper = { depthAt: (y: number) => number | null; yAt: (d: number) => number | null };

/** Depth ruler: ticks right-aligned at xTicks, labels to their left, depth separators across [gridX, gridX+gridW]. */
export function rulerLayers(o: {
  map: Mapper;
  sy: (y: number) => number;
  imgY: (s: number) => number;
  top: number;
  viewH: number;
  screenPxPerFt: number;
  xTicks: number;
  labelX: number;
  labelW: number;
  grid?: { x: number; w: number };
  yRange: [number, number]; // image rows the ruler covers
}): ReactNode[] {
  const out: ReactNode[] = [];
  const a = Math.max(o.yRange[0], o.imgY(o.top - 100));
  const b = Math.min(o.yRange[1], o.imgY(o.top + o.viewH + 100));
  if (b <= a) return out;
  const dA = o.map.depthAt(a);
  const dB = o.map.depthAt(b);
  if (dA == null || dB == null) return out;
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  const major = steps.find((s) => s * o.screenPxPerFt >= 64) ?? 20000;
  const minor = (major / 5) * o.screenPxPerFt >= 9 ? major / 5 : (major / 2) * o.screenPxPerFt >= 9 ? major / 2 : major;
  for (let d = Math.ceil(dA / minor) * minor; d <= dB; d += minor) {
    const y = o.map.yAt(d);
    if (y == null) continue;
    const s = o.sy(y);
    const isMajor = Math.abs(d / major - Math.round(d / major)) < 1e-6;
    out.push(
      <div
        key={`tk${d}`}
        className="absolute h-px"
        style={{ left: o.xTicks - (isMajor ? 12 : 6), width: isMajor ? 12 : 6, top: s, backgroundColor: isMajor ? "#7C8591" : "#4A515C" }}
      />,
    );
    if (isMajor) {
      out.push(
        <span
          key={`tl${d}`}
          className="tabular absolute -translate-y-1/2 text-right font-mono text-[11px] text-muted"
          style={{ left: o.labelX, width: o.labelW, top: s }}
        >
          {fmt(d)}
        </span>,
      );
      if (o.grid) out.push(<div key={`tg${d}`} className="absolute h-px bg-line" style={{ left: o.grid.x, width: o.grid.w, top: s }} />);
    }
  }
  return out;
}

export type Placed = { r: Interval; mid: number; y: number; lh: number };

/** Labels at their bed's depth, pushed apart only as far as needed; beds too crowded to label are counted. */
export function placeLabels(beds: Interval[], sy: (y: number) => number, selected: number | null, top: number, viewH: number) {
  const placed: Placed[] = [];
  let cursor = -Infinity;
  let hidden = 0;
  for (const r of beds) {
    const a = sy(r.top_y);
    const b = sy(r.bottom_y);
    const mid = (a + b) / 2;
    const lh = b - a >= 36 ? 34 : 18;
    const y = Math.max(mid - lh / 2, cursor + 3);
    if (y + lh / 2 - mid > 46 && r.idx !== selected) {
      if (mid >= top && mid <= top + viewH) hidden++;
      continue;
    }
    placed.push({ r, mid, y, lh });
    cursor = y + lh;
  }
  return { placed, hidden };
}

/** Leader lines from the beds (at xFrom) to their labels (at xLabel), and the label buttons. */
export function labelLayers(o: {
  placed: Placed[];
  xFrom: number;
  xLabel: number;
  W: number;
  H: number;
  selected: number | null;
  onSelect: (idx: number) => void;
  onHover: (idx: number | null) => void;
  leader?: string; // leader-line colour
}): ReactNode[] {
  const out: ReactNode[] = [
    <svg key="leaders" className="pointer-events-none absolute left-0 top-0" width={o.W} height={o.H} aria-hidden="true">
      {o.placed.map(({ r, mid, y, lh }) => {
        const lm = y + lh / 2;
        const x2 = o.xLabel - 5;
        const elbow = Math.min(o.xFrom + 10, x2 - 10);
        return (
          <path
            key={r.idx}
            d={`M${o.xFrom},${mid} L${elbow},${mid} L${x2 - 8},${lm} L${x2},${lm}`}
            fill="none"
            stroke={r.idx === o.selected ? SELECT : (o.leader ?? "#4A515C")}
            strokeWidth={1}
          />
        );
      })}
    </svg>,
  ];
  for (const { r, y, lh } of o.placed) {
    const isSel = r.idx === o.selected;
    out.push(
      <button
        key={`l${r.idx}`}
        type="button"
        tabIndex={-1}
        onClick={(e) => {
          e.stopPropagation();
          o.onSelect(r.idx);
        }}
        onMouseEnter={() => o.onHover(r.idx)}
        onMouseLeave={() => o.onHover(null)}
        className={`absolute flex max-w-[340px] flex-col justify-center rounded-md px-2 text-left transition-colors hover:bg-hover ${isSel ? "bg-panel-2" : ""}`}
        style={{ left: o.xLabel, top: y, height: lh, zIndex: 4 }}
      >
        <span className="flex items-center gap-2 whitespace-nowrap text-[13px] leading-4">
          <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-[2px] ${texture(r)}`} style={{ backgroundColor: r.color }} />
          <span className="font-mono text-[11px] text-faint">#{r.idx + 1}</span>
          <span className={`${r.unresolved ? "text-rose" : "text-ink"} ${isSel ? "font-semibold" : ""}`}>{bedName(r)}</span>
          {lh < 30 && <span className="font-mono text-[11px] text-faint">{fmt(r.top_depth, 1)} ft</span>}
        </span>
        {lh >= 30 && (
          <span className="whitespace-nowrap pl-4 font-mono text-[11px] leading-4 text-faint">
            {fmt(r.top_depth, 1)} – {fmt(r.bottom_depth, 1)} ft{r.cluster_id != null ? ` · c${r.cluster_id}` : ""}
          </span>
        )}
      </button>,
    );
  }
  return out;
}

export function HiddenNote({ n }: { n: number }) {
  if (n <= 0) return null;
  return (
    <div className="pointer-events-none absolute bottom-3 left-1/2 z-20 -translate-x-1/2 rounded-full border border-line-2 bg-panel px-3 py-1 text-xs text-muted shadow-lg shadow-black/40">
      {n} thin {n === 1 ? "bed" : "beds"} without a label here · zoom in
    </div>
  );
}
