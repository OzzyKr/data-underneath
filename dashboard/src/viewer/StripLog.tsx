import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Interval, LogDetail } from "../api";
import { fmt } from "../format";
import { depthMapper, depthTicks, shortDepth } from "./depth";
import { HiddenNote, SELECT, estimatePxPerFt, labelLayers, placeLabels, rulerLayers } from "./layers";

/** Feet of log per screen height, or the whole log. */
export type Span = 50 | 200 | 1000 | "all";

type Props = {
  log: LogDetail;
  intervals: Interval[];
  selected: number | null;
  onSelect: (idx: number) => void;
  span: Span;
  jump: { depth: number; n: number } | null;
  onView: (top: number | null, bottom: number | null) => void;
};

export const PAD = 44; // above / below the column (room for the lane header)
const LEFT = 20; // left margin of the track group
const RULER = 72; // depth ruler lane
const PRED = 16; // predicted-rock track
const GAP = 34; // between the tracks and the labels (leader lines run here)
const SCAN_LINE = "#2F5BD3"; // boundaries drawn over the white scan

/** The window (top depth) with the most distinct rocks, then the most beds - like the labelling kit. */
export function bestWindow(intervals: Interval[], windowFt: number): number | null {
  const ivs = intervals.filter((r) => r.top_depth != null && r.bottom_depth != null);
  if (!ivs.length) return null;
  const lo = ivs[0].top_depth!;
  const hi = ivs[ivs.length - 1].bottom_depth!;
  if (hi - lo <= windowFt) return lo;
  let best = lo;
  let bestScore = [-1, -1];
  const step = Math.max(5, windowFt / 10);
  for (let top = lo; top + windowFt <= hi; top += step) {
    const inside = ivs.filter((r) => r.bottom_depth! > top && r.top_depth! < top + windowFt);
    const rocks = new Set(inside.filter((r) => !r.unresolved).map((r) => r.rock)).size;
    const score = [rocks, inside.length];
    if (score[0] > bestScore[0] || (score[0] === bestScore[0] && score[1] > bestScore[1])) {
      best = top;
      bestScore = score;
    }
  }
  return best;
}

/** The lithology column, a predicted-rock track, a depth ruler and every bed labelled at its depth. */
export default function StripLog({ log, intervals, selected, onSelect, span, jump, onView }: Props) {
  const col = log.tiles!.col!;
  const el = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 700 });
  const [measured, setMeasured] = useState(false); // first scroll waits for the real viewer size
  const [top, setTop] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const centre = useRef<number | null>(null);
  const placedFor = useRef<string | null>(null);
  const map = useMemo(() => depthMapper(intervals), [intervals]);
  const pxPerFt = useMemo(() => estimatePxPerFt(log, intervals), [log, intervals]);

  useLayoutEffect(() => {
    const node = el.current;
    if (!node) return;
    const ro = new ResizeObserver(() => setSize({ w: node.clientWidth, h: node.clientHeight }));
    ro.observe(node);
    setSize({ w: node.clientWidth, h: node.clientHeight });
    setMeasured(true);
    return () => ro.disconnect();
  }, []);

  // ---------------------------------------------------------------- geometry
  const Y0 = col.box[1];
  const Y1 = col.box[3];
  const k = span === "all" ? Math.max(1e-4, (size.h - 2 * PAD) / (Y1 - Y0)) : size.h / (span * pxPerFt);
  const H = (Y1 - Y0) * k + 2 * PAD;
  const sy = useCallback((y: number) => PAD + (y - Y0) * k, [Y0, k]);
  const imgY = useCallback((s: number) => Y0 + (s - PAD) / k, [Y0, k]);

  const scanW = Math.round(Math.min(220, Math.max(96, (col.box[2] - col.box[0]) * k)));
  const xRuler = LEFT;
  const xScan = xRuler + RULER;
  const xPred = xScan + scanW + 3;
  const xLabel = xPred + PRED + GAP;
  const W = Math.max(size.w, xLabel + 300);

  // ---------------------------------------------------------------- scrolling
  const scrollTo = (screenY: number) => {
    const node = el.current;
    if (!node) return;
    node.scrollTop = Math.max(0, screenY);
    setTop(node.scrollTop);
    centre.current = imgY(node.scrollTop + node.clientHeight / 2);
  };

  const frame = useRef(0);
  const onScroll = () => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const node = el.current;
      if (!node) return;
      setTop(node.scrollTop);
      centre.current = imgY(node.scrollTop + node.clientHeight / 2);
    });
  };

  // first view of a log: the selected bed, else the most varied part of the log
  useLayoutEffect(() => {
    if (!measured || placedFor.current === log.log_id || size.h < 100) return;
    placedFor.current = log.log_id;
    const sel = intervals.find((r) => r.idx === selected);
    if (sel) {
      scrollTo(sy((sel.top_y + sel.bottom_y) / 2) - size.h / 2);
      return;
    }
    const d = span === "all" ? null : bestWindow(intervals, span);
    const y = d != null ? map.yAt(d) : null;
    scrollTo(y != null ? sy(y) - 24 : 0);
  }, [log.log_id, size.h, intervals, measured]); // eslint-disable-line react-hooks/exhaustive-deps

  // when the zoom changes, land on the focus: the selected bed if there is one, else the same depth as before
  useLayoutEffect(() => {
    const focus = intervals.find((r) => r.idx === selected);
    if (focus) {
      scrollTo(sy((focus.top_y + focus.bottom_y) / 2) - size.h / 2);
      return;
    }
    if (centre.current == null) return;
    scrollTo(sy(centre.current) - size.h / 2);
  }, [k]); // eslint-disable-line react-hooks/exhaustive-deps

  // bring a bed chosen elsewhere (list, CSV, arrows) into view
  useEffect(() => {
    const node = el.current;
    const iv = intervals.find((r) => r.idx === selected);
    if (!node || !iv || placedFor.current !== log.log_id) return;
    const a = sy(iv.top_y);
    const b = sy(iv.bottom_y);
    if (a < node.scrollTop + 30 || b > node.scrollTop + node.clientHeight - 30) scrollTo((a + b) / 2 - node.clientHeight / 2);
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!jump) return;
    const y = map.yAt(jump.depth);
    if (y == null) return;
    scrollTo(sy(y) - size.h / 2);
    const hit = intervals.find((r) => r.top_y <= y && y < r.bottom_y);
    if (hit) onSelect(hit.idx);
  }, [jump?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    onView(map.depthAt(imgY(top)), map.depthAt(imgY(top + size.h)));
  }, [top, size.h, k, map]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------- drawing
  const lo = top - 300;
  const hi = top + size.h + 300;
  const layers: ReactNode[] = [];

  const sel = intervals.find((r) => r.idx === selected);
  if (sel) {
    layers.push(
      <div
        key="selband"
        className="absolute"
        style={{ left: 0, width: W, top: sy(sel.top_y), height: Math.max(1, (sel.bottom_y - sel.top_y) * k), backgroundColor: "rgba(143,116,234,0.10)" }}
      />,
    );
  }

  // the scan: the lithology column at full resolution
  const tileImgH = col.tile_h / col.scale;
  for (let i = 0; i < col.n; i++) {
    const a = Y0 + i * tileImgH;
    const b = Math.min(Y1, a + tileImgH);
    if (sy(b) < lo || sy(a) > hi) continue;
    layers.push(
      <img
        key={`c${i}`}
        src={`/tiles/${log.log_id}/c${String(i).padStart(4, "0")}.webp`}
        alt=""
        draggable={false}
        className="absolute max-w-none select-none"
        style={{ left: xScan, top: sy(a), width: scanW, height: Math.max(1, (b - a) * k) }}
      />,
    );
  }

  layers.push(
    ...rulerLayers({
      map,
      sy,
      imgY,
      top,
      viewH: size.h,
      screenPxPerFt: k * pxPerFt,
      xTicks: xScan,
      labelX: xRuler,
      labelW: RULER - 18,
      grid: { x: xPred + PRED + 2, w: W - xPred - PRED - 2 },
      yRange: [Y0, Y1],
    }),
  );

  // beds: predicted track, boundaries, hit areas
  const vis = intervals.filter((r) => sy(r.bottom_y) >= lo && sy(r.top_y) <= hi);
  let lastBoundary = -Infinity;
  for (const r of vis) {
    const a = sy(r.top_y);
    const h = Math.max(1, sy(r.bottom_y) - a);
    const isSel = r.idx === selected;
    layers.push(
      <div key={`p${r.idx}`} className={`absolute ${r.unresolved ? "hatch" : ""}`} style={{ left: xPred, width: PRED, top: a, height: h, backgroundColor: r.color }} />,
    );
    if (a - lastBoundary >= 3) {
      lastBoundary = a;
      layers.push(
        <div key={`bl${r.idx}`} className="absolute h-px" style={{ left: xScan - 4, width: 4, top: a, backgroundColor: "#C9CED6" }} />,
        <div key={`bs${r.idx}`} className="absolute h-px" style={{ left: xScan, width: scanW, top: a, backgroundColor: SCAN_LINE, opacity: 0.85 }} />,
        <div key={`bp${r.idx}`} className="absolute h-px" style={{ left: xPred, width: PRED + 4, top: a, backgroundColor: "#121419" }} />,
      );
    }
    layers.push(
      <button
        key={`h${r.idx}`}
        type="button"
        aria-label={`Bed ${r.idx + 1}: ${r.rock}, ${fmt(r.top_depth, 1)} to ${fmt(r.bottom_depth, 1)} ft`}
        onClick={() => onSelect(r.idx)}
        onMouseEnter={() => setHover(r.idx)}
        onMouseLeave={() => setHover((v) => (v === r.idx ? null : v))}
        className="absolute cursor-pointer rounded-[2px] outline-none"
        style={{
          left: xScan - 2,
          width: xPred + PRED - xScan + 4,
          top: a,
          height: h,
          boxShadow: isSel ? `0 0 0 2px ${SELECT}` : r.idx === hover ? `0 0 0 1px ${SELECT}` : undefined,
          zIndex: isSel ? 3 : 2,
        }}
      />,
    );
  }

  const { placed, hidden } = placeLabels(vis, sy, selected, top, size.h);
  layers.push(...labelLayers({ placed, xFrom: xPred + PRED + 3, xLabel, W, H, selected, onSelect, onHover: setHover }));

  const head = (x: number, w: number, text: string, align: "left" | "center" | "right" = "center") => (
    <span key={text} className="absolute truncate text-[10px] uppercase tracking-wider text-faint" style={{ left: x, width: w, top: 9, textAlign: align }}>
      {text}
    </span>
  );

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1">
      <div className="relative min-w-0 flex-1">
        <div
          ref={el}
          onScroll={onScroll}
          tabIndex={0}
          aria-label="Lithology column"
          onKeyDown={(e) => {
            if (selected == null) return;
            if (e.key === "ArrowDown" && selected < intervals.length - 1) {
              e.preventDefault();
              onSelect(selected + 1);
            }
            if (e.key === "ArrowUp" && selected > 0) {
              e.preventDefault();
              onSelect(selected - 1);
            }
          }}
          className="absolute inset-0 overflow-auto outline-none"
        >
          <div className="relative" style={{ width: W, height: H }}>
            {layers}
          </div>
        </div>
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-8 overflow-hidden border-b border-line bg-nav/95">
          {head(xRuler, RULER - 18, "Depth ft", "right")}
          {head(xScan - 10, scanW + 20, "Scan")}
          {head(xPred - 14, PRED + 28, "Pred.")}
          {head(xLabel + 8, 200, "Beds", "left")}
        </div>
        <HiddenNote n={hidden} />
      </div>
      <Minimap
        intervals={intervals}
        y0={Y0}
        y1={Y1}
        height={size.h}
        top={top}
        contentH={H}
        depthAt={map.depthAt}
        yAt={map.yAt}
        onSeek={(frac) => scrollTo(frac * H - size.h / 2)}
      />
    </div>
  );
}

export function Minimap({
  intervals,
  y0,
  y1,
  height,
  top,
  contentH,
  depthAt,
  yAt,
  onSeek,
}: {
  intervals: Interval[];
  y0: number;
  y1: number;
  height: number;
  top: number;
  contentH: number;
  depthAt: (y: number) => number | null;
  yAt: (d: number) => number | null;
  onSeek: (frac: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const span = y1 - y0;
  const pad = PAD;
  const mh = Math.max(50, height - 2 * pad);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = 12 * dpr;
    c.height = mh * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.fillStyle = "#1C2027";
    g.fillRect(0, 0, 12, mh);
    for (const r of intervals) {
      const a = ((r.top_y - y0) / span) * mh;
      const b = ((r.bottom_y - y0) / span) * mh;
      g.fillStyle = r.unresolved ? "#4A515C" : r.color;
      g.fillRect(0, a, 12, Math.max(0.6, b - a));
    }
  }, [intervals, y0, span, mh]);

  const dTop = depthAt(y0);
  const dBot = depthAt(y1);
  const ticks = dTop != null && dBot != null ? depthTicks(dTop, dBot, Math.max(3, Math.floor(mh / 90))) : [];
  const frac = (s: number) => (s / Math.max(1, contentH)) * mh;

  const seek = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)));
  };

  return (
    <div className="relative w-[70px] shrink-0 border-l border-line">
      <span className="absolute inset-x-0 top-0 h-8 border-b border-line pt-[9px] text-center text-[10px] uppercase tracking-wider text-faint">Log</span>
      {ticks.map((dp) => {
        const y = yAt(dp);
        if (y == null || y < y0 || y > y1) return null;
        return (
          <span key={dp} className="absolute left-1.5 -translate-y-1/2 font-mono text-[10px] text-faint" style={{ top: pad + ((y - y0) / span) * mh }}>
            {shortDepth(dp)}
          </span>
        );
      })}
      <div
        role="slider"
        tabIndex={0}
        aria-label="Position in the log"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round((top / Math.max(1, contentH)) * 100)}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seek(e);
        }}
        onPointerMove={(e) => e.buttons === 1 && seek(e)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") onSeek(Math.min(1, (top + height * 1.5) / contentH));
          if (e.key === "ArrowUp") onSeek(Math.max(0, (top - height * 0.5) / contentH));
        }}
        className="absolute right-2.5 cursor-pointer"
        style={{ top: pad, width: 12, height: mh }}
      >
        <canvas ref={canvas} className="block rounded-sm" style={{ width: 12, height: mh }} />
        <div className="pointer-events-none absolute -left-1 -right-1 rounded-sm border-[1.5px] border-ink" style={{ top: frac(top), height: Math.max(6, frac(height)) }} />
      </div>
    </div>
  );
}
