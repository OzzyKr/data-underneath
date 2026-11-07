import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Interval, LogDetail } from "../api";
import { texture } from "../components/ui";
import { fmt } from "../format";
import { depthMapper } from "./depth";
import { HiddenNote, SELECT, estimatePxPerFt, labelLayers, placeLabels, rulerLayers } from "./layers";
import { Minimap, PAD } from "./StripLog";

export type Fit = "width" | "page";

type Props = {
  log: LogDetail;
  intervals: Interval[];
  selected: number | null;
  onSelect: (idx: number) => void;
  fit: Fit;
  spotlight: boolean;
  onOpen: (depth: number) => void; // open the column view at this depth
};

const LEFT = 20;
const RULER = 72;
const GAP = 34;
const LABELS = 260;
const BOTTOM = 24;

type BoxKey = "depth_column" | "interpreted_lithology_column" | "lithology_legend";
export const REGIONS: { key: BoxKey; name: string; color: string }[] = [
  { key: "lithology_legend", name: "Legend", color: "#E79A3C" },
  { key: "depth_column", name: "Depth column", color: "#E5484D" },
  { key: "interpreted_lithology_column", name: "Lithology column", color: "#3987E5" },
];

/** The original scan, uncropped and true to proportion, with what the pipeline read and found drawn on it. */
export default function PageView({ log, intervals, selected, onSelect, fit, spotlight, onOpen }: Props) {
  const p = log.tiles!.page!;
  const b = log.boxes!;
  const el = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 1100, h: 700 });
  const [measured, setMeasured] = useState(false);
  const [top, setTop] = useState(0);
  const [hoverY, setHoverY] = useState<number | null>(null);
  const [hoverBedIdx, setHoverBedIdx] = useState<number | null>(null);
  const placed0 = useRef<string | null>(null);
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

  // ---------------------------------------------------------------- geometry (the page keeps its proportions)
  const imgH = p.image[1];
  const availW = Math.max(240, size.w - LEFT - RULER - GAP - LABELS - 16);
  const byWidth = availW / p.w;
  const d = fit === "width" ? byWidth : Math.min(byWidth, (size.h - PAD - BOTTOM) / p.h);
  const s = p.scale * d; // screen px per image px
  const W = p.w * d;
  const H = p.h * d;
  const xPage = LEFT + RULER;
  const xLabel = xPage + W + GAP;
  const CW = Math.max(size.w, xLabel + LABELS + 20);
  const CH = PAD + H + BOTTOM;
  const X = (x: number) => xPage + x * s;
  const Y = useCallback((y: number) => PAD + y * s, [s]);
  const imgY = useCallback((sc: number) => (sc - PAD) / s, [s]);
  const lc = b.interpreted_lithology_column;

  const scrollTo = (screenY: number) => {
    const node = el.current;
    if (!node) return;
    node.scrollTop = Math.max(0, screenY);
    setTop(node.scrollTop);
  };

  // open on the selected bed, else at the top of the page
  useLayoutEffect(() => {
    if (!measured || placed0.current === `${log.log_id}:${fit}` || size.h < 100) return;
    placed0.current = `${log.log_id}:${fit}`;
    const sel = intervals.find((r) => r.idx === selected);
    scrollTo(sel ? Y((sel.top_y + sel.bottom_y) / 2) - size.h / 2 : 0);
  }, [log.log_id, fit, size.h, measured]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const node = el.current;
    const iv = intervals.find((r) => r.idx === selected);
    if (!node || !iv || placed0.current == null) return;
    const a = Y(iv.top_y);
    const c = Y(iv.bottom_y);
    if (a < node.scrollTop + 40 || c > node.scrollTop + node.clientHeight - 20) scrollTo((a + c) / 2 - node.clientHeight / 2);
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
  layers.push(<div key="paper" className="absolute shadow-2xl shadow-black/60" style={{ left: xPage, top: PAD, width: W, height: H, background: "#FBFBF9" }} />);
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
        style={{ left: xPage, top: Y(a), width: W, height: Math.max(1, (c - a) * s) }}
      />,
    );
  }

  // spotlight: dim the page except the three regions the pipeline reads
  if (spotlight) {
    const hole = (k: BoxKey) => {
      const [x0, y0, x1, y1] = b[k];
      const w = (x1 - x0) * s + 6;
      const h = (y1 - y0) * s + 6;
      return `M${X(x0) - 3},${Y(y0) - 3} h${w} v${h} h${-w} Z`;
    };
    const page = `M${xPage},${PAD} h${W} v${H} h${-W} Z`;
    layers.push(
      <svg key="spot" className="pointer-events-none absolute left-0 top-0" width={CW} height={CH} aria-hidden="true">
        <path d={[page, ...REGIONS.map((r) => hole(r.key))].join(" ")} fillRule="evenodd" fill="rgba(14,16,20,0.62)" />
      </svg>,
    );
  }

  // predicted beds on the lithology column, with their boundaries
  const vis = intervals.filter((r) => Y(r.bottom_y) >= lo && Y(r.top_y) <= hi);
  let lastBoundary = -Infinity;
  const lcW = (lc[2] - lc[0]) * s;
  for (const r of vis) {
    const a = Y(r.top_y);
    const h = Math.max(0.75, Y(r.bottom_y) - a);
    layers.push(
      <div
        key={`f${r.idx}`}
        className={`pointer-events-none absolute ${texture(r)}`}
        style={{ left: X(lc[0]), width: Math.max(3, lcW * 0.5), top: a, height: h, backgroundColor: r.color, opacity: 0.85 }}
      />,
    );
    if (a - lastBoundary >= 3) {
      lastBoundary = a;
      layers.push(<div key={`b${r.idx}`} className="pointer-events-none absolute h-px" style={{ left: X(lc[0]), width: lcW, top: a, backgroundColor: "#2F5BD3" }} />);
    }
  }

  // the three regions, outlined and named. Tags that would overlap stack one row apart.
  const tagWidth = (r: (typeof REGIONS)[number]) => r.name.length * 6.3 + 14;
  const tagLift: Partial<Record<BoxKey, number>> = {};
  {
    const dcR = REGIONS.find((r) => r.key === "depth_column")!;
    const [dx0, dy0] = b.depth_column;
    const [lx0, ly0] = b.interpreted_lithology_column;
    const overlapX = X(dx0) < X(lx0) ? X(dx0) + tagWidth(dcR) > X(lx0) : X(lx0) + tagWidth(REGIONS[2]) > X(dx0);
    if (overlapX && Math.abs(Y(dy0) - Y(ly0)) < 24) tagLift.depth_column = 22;
  }
  for (const r of REGIONS) {
    const [x0, y0, x1, y1] = b[r.key];
    layers.push(
      <div
        key={r.key}
        className="pointer-events-none absolute rounded-[2px] border-2"
        style={{ left: X(x0) - 1, top: Y(y0) - 1, width: (x1 - x0) * s + 2, height: (y1 - y0) * s + 2, borderColor: r.color }}
      />,
    );
    const pinned = Y(y0) < top + PAD + 6;
    if (Y(y1) > top + PAD + 30 && Y(y0) < top + size.h) {
      const lift = tagLift[r.key] ?? 0;
      const tagTop = pinned ? Math.min(top + PAD + 6 + lift, Y(y1) - 24) : Y(y0) - 3 - lift;
      layers.push(
        <span
          key={`${r.key}-tag`}
          className="pointer-events-none absolute z-[5] whitespace-nowrap rounded-[3px] px-1.5 py-0.5 text-[11px] text-white shadow-md"
          style={{
            left: X(x0) - 1,
            top: tagTop,
            backgroundColor: r.color,
            transform: pinned ? undefined : "translateY(-100%)",
          }}
        >
          <span className="font-semibold">{r.name}</span>
        </span>,
      );
    }
  }

  // selected bed
  const sel = intervals.find((r) => r.idx === selected);
  if (sel) {
    layers.push(
      <div
        key="sel"
        className="pointer-events-none absolute rounded-[2px]"
        style={{ left: X(lc[0]) - 3, width: lcW + 6, top: Y(sel.top_y), height: Math.max(3, (sel.bottom_y - sel.top_y) * s), boxShadow: `0 0 0 2px ${SELECT}`, zIndex: 3 }}
      />,
    );
  }

  // depth ruler on the left of the page, separators across the label lane
  layers.push(
    ...rulerLayers({
      map,
      sy: Y,
      imgY,
      top,
      viewH: size.h,
      screenPxPerFt: s * pxPerFt,
      xTicks: xPage - 2,
      labelX: LEFT,
      labelW: RULER - 20,
      grid: { x: xPage + W + 2, w: CW - xPage - W - 2 },
      yRange: [lc[1], lc[3]],
    }),
  );

  // labels at their depth, leaders from the lithology column across the page
  const { placed, hidden } = placeLabels(vis, Y, selected, top, size.h);
  layers.push(...labelLayers({ placed, xFrom: X(lc[2]) + 2, xLabel, W: CW, H: CH, selected, onSelect, onHover: setHoverBedIdx, leader: "#9AA3AD" }));

  // hover: a line across the page with the depth; a click opens the column view there
  const inLog = hoverY != null && hoverY >= lc[1] && hoverY <= lc[3];
  const hoverDepth = inLog ? map.depthAt(hoverY!) : null;
  if (inLog && hoverDepth != null && hoverBedIdx == null) {
    layers.push(
      <div key="guide" className="pointer-events-none absolute z-[6] h-px bg-select" style={{ left: xPage - 8, width: W + 16, top: Y(hoverY!) }} />,
      <span
        key="guide-tag"
        className="tabular pointer-events-none absolute z-[6] -translate-y-1/2 rounded bg-select px-1.5 py-0.5 font-mono text-[11px] text-white"
        style={{ right: CW - xPage + 10, top: Y(hoverY!) }}
      >
        {fmt(hoverDepth, 1)}
      </span>,
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
            const x = e.clientX - rect.left + e.currentTarget.scrollLeft;
            setHoverY(x >= xPage && x <= xPage + W ? imgY(e.clientY - rect.top + e.currentTarget.scrollTop) : null);
          }}
          onMouseLeave={() => setHoverY(null)}
          onClick={() => {
            if (inLog && hoverDepth != null) onOpen(hoverDepth);
          }}
          style={{ cursor: inLog ? "zoom-in" : "default" }}
        >
          <div className="relative" style={{ width: CW, height: CH }}>
            {layers}
          </div>
        </div>
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex h-8 items-center overflow-hidden border-b border-line bg-nav/95 text-[10px] uppercase tracking-wider text-faint">
          <span className="absolute whitespace-nowrap text-right" style={{ left: LEFT - 8, width: RULER - 12 }}>
            Depth ft
          </span>
          <span className="absolute flex gap-3 normal-case tracking-normal" style={{ left: xPage }}>
            {REGIONS.map((r) => (
              <span key={r.key} className="flex items-center gap-1.5 text-[11px] text-muted">
                <span className="h-2.5 w-2.5 rounded-[2px] border-2" style={{ borderColor: r.color }} />
                {r.name}
              </span>
            ))}
          </span>
          <span className="absolute" style={{ left: xLabel + 8 }}>
            Beds
          </span>
        </div>
        <HiddenNote n={hidden} />
      </div>
      <Minimap
        intervals={intervals}
        y0={0}
        y1={imgH}
        height={size.h}
        top={top}
        contentH={CH}
        depthAt={map.depthAt}
        yAt={map.yAt}
        onSeek={(frac) => scrollTo(frac * CH - size.h / 2)}
      />
    </div>
  );
}
