import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Interval, LogDetail } from "../api";
import { fmt } from "../format";
import { depthMapper, depthTicks, shortDepth } from "./depth";

export type Zoom = "fit" | "100" | "200";

type Props = {
  log: LogDetail;
  intervals: Interval[];
  selected: number | null;
  onSelect: (idx: number) => void;
  zoom: Zoom;
  jump: { depth: number; n: number } | null;
  onView: (top: number | null, bottom: number | null) => void;
};

const GUTTER = 72; // room left of the scan for depth labels
const RIGHT = 120; // room right of the scan for rock labels
const LINE = "#1F4FA8";
const MIN_LABEL_GAP = 15;

export default function StripViewer({ log, intervals, selected, onSelect, zoom, jump, onView }: Props) {
  const t = log.tiles!;
  const b = log.boxes!;
  const el = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 640 });
  const [top, setTop] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const centre = useRef<number | null>(null); // image row at the viewport centre, kept across zoom changes
  const map = useMemo(() => depthMapper(intervals), [intervals]);

  useLayoutEffect(() => {
    const node = el.current;
    if (!node) return;
    const ro = new ResizeObserver(() => setSize({ w: node.clientWidth, h: node.clientHeight }));
    ro.observe(node);
    setSize({ w: node.clientWidth, h: node.clientHeight });
    return () => ro.disconnect();
  }, []);

  const d = zoom === "fit" ? Math.max(0.2, Math.min(1.5, (size.w - GUTTER - RIGHT) / t.w)) : zoom === "100" ? 1 : 2;
  const k = t.scale * d; // screen px per image px
  const W = t.w * d + GUTTER + RIGHT;
  const H = t.h * d;
  const X = useCallback((x: number) => GUTTER + (x - t.view[0]) * k, [k, t.view]);
  const Y = useCallback((y: number) => (y - t.view[1]) * k, [k, t.view]);
  const imgY = useCallback((screenY: number) => t.view[1] + screenY / k, [k, t.view]);

  // keep the same place in view when the zoom changes
  useLayoutEffect(() => {
    const node = el.current;
    if (!node || centre.current == null) return;
    node.scrollTop = Y(centre.current) - node.clientHeight / 2;
    setTop(node.scrollTop);
  }, [k]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // report the visible depth range
  useEffect(() => {
    onView(map.depthAt(imgY(top)), map.depthAt(imgY(top + size.h)));
  }, [top, size.h, k, map]); // eslint-disable-line react-hooks/exhaustive-deps

  // bring the selected bed into view
  useEffect(() => {
    const node = el.current;
    const iv = intervals.find((r) => r.idx === selected);
    if (!node || !iv) return;
    const y0 = Y(iv.top_y);
    const y1 = Y(iv.bottom_y);
    if (y0 < node.scrollTop + 40 || y1 > node.scrollTop + node.clientHeight - 40) {
      node.scrollTop = Math.max(0, y0 - node.clientHeight * 0.35);
      setTop(node.scrollTop);
    }
  }, [selected, intervals, k]); // eslint-disable-line react-hooks/exhaustive-deps

  // "go to depth"
  useEffect(() => {
    const node = el.current;
    if (!node || !jump) return;
    const y = map.yAt(jump.depth);
    if (y == null) return;
    node.scrollTop = Math.max(0, Y(y) - node.clientHeight / 2);
    setTop(node.scrollTop);
    const hit = intervals.find((r) => r.top_y <= y && y < r.bottom_y);
    if (hit) onSelect(hit.idx);
  }, [jump?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  const th = t.tile_h * d;
  const k0 = Math.max(0, Math.floor(top / th) - 1);
  const k1 = Math.min(t.n - 1, Math.floor((top + size.h) / th) + 1);
  const tiles = [];
  for (let i = k0; i <= k1; i++) {
    const h = (i < t.n - 1 ? t.tile_h : t.h - i * t.tile_h) * d;
    tiles.push(
      <img
        key={i}
        src={`/tiles/${log.log_id}/t${String(i).padStart(4, "0")}.webp`}
        alt=""
        draggable={false}
        className="absolute select-none"
        style={{ left: GUTTER, top: i * th, width: t.w * d, height: h }}
      />,
    );
  }

  const dc = b.depth_column;
  const lc = b.interpreted_lithology_column;
  const lo = top - 400;
  const hi = top + size.h + 400;
  const visible = intervals.filter((r) => Y(r.bottom_y) >= lo && Y(r.top_y) <= hi);

  const overlay = [];
  let lastDepth = -Infinity;
  let lastRock = -Infinity;
  for (const r of visible) {
    const y0 = Y(r.top_y);
    const y1 = Y(r.bottom_y);
    const h = Math.max(1, y1 - y0);
    const isSel = r.idx === selected;
    const isHover = r.idx === hover;
    overlay.push(
      <div
        key={`f${r.idx}`}
        className={`pointer-events-none absolute ${r.unresolved ? "hatch" : ""}`}
        style={{ left: X(lc[0]), top: y0, width: (lc[2] - lc[0]) * k * 0.5, height: h, backgroundColor: r.color, opacity: 0.6 }}
      />,
      <div
        key={`b${r.idx}`}
        className="pointer-events-none absolute"
        style={{ left: X(dc[0]), top: y0, width: X(lc[2]) - X(dc[0]), height: 1, backgroundColor: LINE, opacity: 0.75 }}
      />,
      <button
        key={`h${r.idx}`}
        type="button"
        aria-label={`${r.rock}, ${fmt(r.top_depth, 2)} to ${fmt(r.bottom_depth, 2)} ft`}
        onClick={() => onSelect(r.idx)}
        onMouseEnter={() => setHover(r.idx)}
        onMouseLeave={() => setHover((v) => (v === r.idx ? null : v))}
        className="absolute cursor-pointer rounded-[2px]"
        style={{
          left: X(lc[0]) - 3,
          top: y0,
          width: X(lc[2]) - X(lc[0]) + 6,
          height: h,
          boxShadow: isSel ? "0 0 0 2px #8F74EA" : isHover ? "0 0 0 1px #8F74EA" : undefined,
          zIndex: isSel ? 3 : isHover ? 2 : 1,
        }}
      />,
    );
    if (y0 - lastDepth >= MIN_LABEL_GAP && r.top_depth != null) {
      lastDepth = y0;
      overlay.push(
        <span
          key={`d${r.idx}`}
          className="pointer-events-none absolute -translate-y-1/2 rounded-sm bg-paper/90 px-1 text-right font-mono text-[10px] leading-4"
          style={{ right: W - X(dc[0]) + 4, top: y0, color: LINE }}
        >
          {fmt(r.top_depth, 1)}
        </span>,
      );
    }
    const mid = (y0 + y1) / 2;
    if ((h >= 18 || isSel) && mid - lastRock >= MIN_LABEL_GAP) {
      lastRock = mid;
      overlay.push(
        <span
          key={`r${r.idx}`}
          className="pointer-events-none absolute -translate-y-1/2 whitespace-nowrap rounded-sm bg-paper/90 px-1 text-[11px] leading-4 text-[#2A2F36]"
          style={{ left: X(lc[2]) + 6, top: mid, fontWeight: isSel ? 600 : 400 }}
        >
          {r.rock}
          {r.cluster_id != null && <span className="text-[#5B6470]"> · c{r.cluster_id}</span>}
        </span>,
      );
    }
  }
  const hv = intervals.find((r) => r.idx === hover);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1">
      <div ref={el} onScroll={onScroll} className="relative min-w-0 flex-1 overflow-auto">
        <div className="relative mx-auto" style={{ width: W, height: H }}>
          {tiles}
          <div
            className="pointer-events-none absolute border border-[#C0392B]"
            style={{ left: X(dc[0]), top: Y(dc[1]), width: (dc[2] - dc[0]) * k, height: (dc[3] - dc[1]) * k }}
          />
          <div
            className="pointer-events-none absolute border border-[#2E86C1]"
            style={{ left: X(lc[0]), top: Y(lc[1]), width: (lc[2] - lc[0]) * k, height: (lc[3] - lc[1]) * k }}
          />
          {overlay}
          {hv && hv.idx !== selected && (
            <div
              className="pointer-events-none absolute z-10 -translate-y-full whitespace-nowrap rounded-md border border-line-2 bg-panel px-2.5 py-1.5 text-xs text-ink shadow-lg"
              style={{ left: X(lc[2]) + 6, top: Y(hv.top_y) - 4 }}
            >
              {hv.rock} · {fmt(hv.top_depth, 1)} – {fmt(hv.bottom_depth, 1)} ft
            </div>
          )}
        </div>
      </div>
      <Minimap
        intervals={intervals}
        view={t.view}
        height={size.h}
        top={top}
        contentH={H}
        depthAt={map.depthAt}
        yAt={map.yAt}
        onSeek={(frac) => {
          const node = el.current;
          if (!node) return;
          node.scrollTop = Math.max(0, frac * H - node.clientHeight / 2);
          setTop(node.scrollTop);
        }}
      />
    </div>
  );
}

function Minimap({
  intervals,
  view,
  height,
  top,
  contentH,
  depthAt,
  yAt,
  onSeek,
}: {
  intervals: Interval[];
  view: [number, number, number, number];
  height: number;
  top: number;
  contentH: number;
  depthAt: (y: number) => number | null;
  yAt: (d: number) => number | null;
  onSeek: (frac: number) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const span = view[3] - view[1];
  const pad = 8;
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
      const y0 = ((r.top_y - view[1]) / span) * mh;
      const y1 = ((r.bottom_y - view[1]) / span) * mh;
      g.fillStyle = r.unresolved ? "#4A515C" : r.color;
      g.fillRect(0, y0, 12, Math.max(0.6, y1 - y0));
    }
  }, [intervals, view, span, mh]);

  const top0 = depthAt(view[1]);
  const bot0 = depthAt(view[3]);
  const ticks = top0 != null && bot0 != null ? depthTicks(top0, bot0, Math.max(3, Math.floor(mh / 90))) : [];
  const vTop = (top / contentH) * mh;
  const vH = Math.max(6, (height / contentH) * mh);

  const seek = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)));
  };

  return (
    <div className="relative w-[68px] shrink-0 border-l border-line" style={{ paddingTop: pad, paddingBottom: pad }}>
      {ticks.map((dp) => {
        const y = yAt(dp);
        if (y == null) return null;
        return (
          <span
            key={dp}
            className="absolute left-1.5 -translate-y-1/2 font-mono text-[10px] text-faint"
            style={{ top: pad + ((y - view[1]) / span) * mh }}
          >
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
        <div
          className="pointer-events-none absolute -left-1 -right-1 rounded-sm border-[1.5px] border-ink"
          style={{ top: vTop, height: vH }}
        />
      </div>
    </div>
  );
}
