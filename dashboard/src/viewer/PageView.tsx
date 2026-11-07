import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Interval, LogDetail } from "../api";
import { fmt } from "../format";
import { depthMapper } from "./depth";
import { Minimap } from "./StripLog";

export type Fit = "width" | "page";

type Props = {
  log: LogDetail;
  intervals: Interval[];
  selected: number | null;
  onSelect: (idx: number) => void;
  fit: Fit;
  focus: [number | null, number | null]; // depth range last shown in the column view
  onOpen: (depth: number) => void; // open the column view at this depth
};

const M = 32; // margin around the page
const BOXES: { key: "depth_column" | "interpreted_lithology_column" | "lithology_legend"; label: string; color: string }[] = [
  { key: "depth_column", label: "Depth column", color: "#E5484D" },
  { key: "interpreted_lithology_column", label: "Lithology column", color: "#3987E5" },
  { key: "lithology_legend", label: "Legend", color: "#E79A3C" },
];

/** The whole scanned page, uncropped, with what the pipeline found drawn on it. */
export default function PageView({ log, intervals, selected, onSelect, fit, focus, onOpen }: Props) {
  const p = log.tiles!.page!;
  const b = log.boxes!;
  const el = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 700 });
  const [measured, setMeasured] = useState(false); // first scroll waits for the real viewer size
  const [top, setTop] = useState(0);
  const [hoverY, setHoverY] = useState<number | null>(null); // image row under the mouse
  const placed = useRef<string | null>(null);
  const map = useMemo(() => depthMapper(intervals), [intervals]);

  useLayoutEffect(() => {
    const node = el.current;
    if (!node) return;
    const ro = new ResizeObserver(() => setSize({ w: node.clientWidth, h: node.clientHeight }));
    ro.observe(node);
    setSize({ w: node.clientWidth, h: node.clientHeight });
    setMeasured(true);
    return () => ro.disconnect();
  }, []);

  const [imgW, imgH] = p.image;
  const byWidth = (size.w - 2 * M) / p.w;
  const d = fit === "width" ? byWidth : Math.min(byWidth, (size.h - 2 * M) / p.h);
  const s = p.scale * d; // screen px per image px
  const W = p.w * d;
  const H = p.h * d;
  const left = Math.max(M, (size.w - W) / 2);
  const X = (x: number) => left + x * s;
  const Y = (y: number) => M + y * s;

  const lc = b.interpreted_lithology_column;
  const focusY0 = focus[0] != null ? map.yAt(focus[0]) : null;
  const focusY1 = focus[1] != null ? map.yAt(focus[1]) : null;

  const scrollTo = (screenY: number) => {
    const node = el.current;
    if (!node) return;
    node.scrollTop = Math.max(0, screenY);
    setTop(node.scrollTop);
  };

  // open on the part last seen in the column view (or the selected bed)
  useLayoutEffect(() => {
    if (!measured || placed.current === `${log.log_id}:${fit}` || size.h < 100) return;
    placed.current = `${log.log_id}:${fit}`;
    const sel = intervals.find((r) => r.idx === selected);
    const y = focusY0 != null && focusY1 != null ? (focusY0 + focusY1) / 2 : sel ? (sel.top_y + sel.bottom_y) / 2 : null;
    scrollTo(y != null ? Y(y) - size.h / 2 : 0);
  }, [log.log_id, fit, size.h, measured]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const node = el.current;
    const iv = intervals.find((r) => r.idx === selected);
    if (!node || !iv || placed.current == null) return;
    const a = Y(iv.top_y);
    const c = Y(iv.bottom_y);
    if (a < node.scrollTop + 20 || c > node.scrollTop + node.clientHeight - 20) scrollTo((a + c) / 2 - node.clientHeight / 2);
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps

  const frame = useRef(0);
  const onScroll = () => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => el.current && setTop(el.current.scrollTop));
  };

  const lo = top - 400;
  const hi = top + size.h + 400;
  const layers: ReactNode[] = [];

  // the page
  const tileImgH = p.tile_h / p.scale;
  for (let i = 0; i < p.n; i++) {
    const a = i * tileImgH;
    const c = Math.min(imgH, a + tileImgH);
    if (Y(c) < lo || Y(a) > hi) continue;
    layers.push(
      <img
        key={`p${i}`}
        src={`/tiles/${log.log_id}/p${String(i).padStart(4, "0")}.webp`}
        alt=""
        draggable={false}
        className="absolute max-w-none select-none"
        style={{ left, top: Y(a), width: W, height: Math.max(1, (c - a) * s) }}
      />,
    );
  }

  // predicted beds on the left half of the lithology column (the organisers' style)
  for (const r of intervals) {
    const a = Y(r.top_y);
    const c = Y(r.bottom_y);
    if (c < lo || a > hi) continue;
    layers.push(
      <div
        key={`f${r.idx}`}
        className={`pointer-events-none absolute ${r.unresolved ? "hatch" : ""}`}
        style={{ left: X(lc[0]), width: Math.max(2, (lc[2] - lc[0]) * s * 0.5), top: a, height: Math.max(0.75, c - a), backgroundColor: r.color, opacity: 0.75 }}
      />,
    );
  }

  // the three boxes
  for (const bx of BOXES) {
    const [x0, y0, x1, y1] = b[bx.key];
    layers.push(
      <div
        key={bx.key}
        className="pointer-events-none absolute rounded-[2px] border-[1.5px]"
        style={{ left: X(x0), top: Y(y0), width: (x1 - x0) * s, height: (y1 - y0) * s, borderColor: bx.color }}
      />,
    );
    // the tag sits on top of its box, or pins to the top of the viewer while the box runs on below
    const pinned = Y(y0) < top + 28;
    const tagTop = pinned ? Math.min(top + 8, Y(y1) - 22) : Y(y0);
    if (Y(y1) > top + 30 && Y(y0) < top + size.h) {
      layers.push(
        <span
          key={`${bx.key}-tag`}
          className="pointer-events-none absolute z-[5] whitespace-nowrap rounded-[3px] px-1.5 py-0.5 text-[10px] font-medium text-white shadow"
          style={{ left: X(x0), top: tagTop, backgroundColor: bx.color, transform: pinned ? undefined : "translateY(-100%)" }}
        >
          {bx.label}
        </span>,
      );
    }
  }

  // where the column view was looking
  if (focusY0 != null && focusY1 != null) {
    layers.push(
      <div
        key="lens"
        className="pointer-events-none absolute border-y-[1.5px] border-select bg-select/10"
        style={{ left: left - 10, width: W + 20, top: Y(focusY0), height: Math.max(4, (focusY1 - focusY0) * s) }}
      >
        <span className="absolute -top-px right-0 -translate-y-full rounded-t-[3px] bg-select px-1.5 py-0.5 text-[10px] font-medium text-white">
          Column view
        </span>
      </div>,
    );
  }

  const sel = intervals.find((r) => r.idx === selected);
  if (sel) {
    layers.push(
      <div
        key="sel"
        className="pointer-events-none absolute rounded-[2px]"
        style={{ left: X(lc[0]) - 3, width: (lc[2] - lc[0]) * s + 6, top: Y(sel.top_y), height: Math.max(3, (sel.bottom_y - sel.top_y) * s), boxShadow: "0 0 0 2px #8F74EA" }}
      />,
    );
  }

  // hover guide: a line across the page with the depth and the bed under it
  const inLog = hoverY != null && hoverY >= lc[1] && hoverY <= lc[3];
  const hoverDepth = inLog ? map.depthAt(hoverY!) : null;
  const hoverBed = inLog ? intervals.find((r) => r.top_y <= hoverY! && hoverY! < r.bottom_y) : undefined;
  if (inLog && hoverDepth != null) {
    layers.push(
      <div key="guide" className="pointer-events-none absolute h-px bg-ink/70" style={{ left: left - 10, width: W + 20, top: Y(hoverY!) }} />,
      <div
        key="guide-tag"
        className="pointer-events-none absolute z-10 flex -translate-y-1/2 items-center gap-2 whitespace-nowrap rounded-md border border-line-2 bg-panel px-2.5 py-1.5 text-xs text-ink shadow-lg shadow-black/40"
        style={{ left: X(lc[2]) + 12, top: Y(hoverY!) }}
      >
        <span className="tabular font-mono">{fmt(hoverDepth, 1)} ft</span>
        {hoverBed && (
          <>
            <span aria-hidden="true" className={`h-2 w-2 rounded-[2px] ${hoverBed.unresolved ? "hatch" : ""}`} style={{ backgroundColor: hoverBed.color }} />
            <span className={hoverBed.unresolved ? "text-rose" : ""}>
              #{hoverBed.idx + 1} {hoverBed.rock}
            </span>
          </>
        )}
        <span className="text-faint">· click to open</span>
      </div>,
    );
  }

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1">
      <div className="relative min-w-0 flex-1">
        <div
          ref={el}
          onScroll={onScroll}
          className="absolute inset-0 overflow-auto"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            setHoverY((e.clientY - rect.top + e.currentTarget.scrollTop - M) / s);
          }}
          onMouseLeave={() => setHoverY(null)}
          onClick={() => {
            if (!inLog || hoverDepth == null) return;
            if (hoverBed) onSelect(hoverBed.idx);
            onOpen(hoverDepth);
          }}
          style={{ cursor: inLog ? "zoom-in" : "default" }}
        >
          <div className="relative" style={{ width: Math.max(size.w, W + 2 * M), height: H + 2 * M }}>
            <div
              className="pointer-events-none absolute shadow-2xl shadow-black/60"
              style={{ left, top: M, width: W, height: H, background: "#FBFBF9" }}
            />
            {layers}
          </div>
        </div>
        <div className="pointer-events-none absolute bottom-3 left-4 z-10 rounded-md bg-nav/85 px-2.5 py-1 text-[11px] text-muted backdrop-blur-sm">
          {fmt(imgW)} × {fmt(imgH)} px · {fmt(intervals.length)} beds
        </div>
      </div>
      <Minimap
        intervals={intervals}
        y0={0}
        y1={imgH}
        height={size.h}
        top={top}
        contentH={H + 2 * M}
        depthAt={map.depthAt}
        yAt={map.yAt}
        onSeek={(frac) => scrollTo(frac * (H + 2 * M) - size.h / 2)}
      />
    </div>
  );
}
