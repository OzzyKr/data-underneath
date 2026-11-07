import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useIntervals, useLog, type Interval, type LogDetail } from "../api";
import { fmt, rememberLog } from "../format";
import { Dot, ErrorNote, bedName } from "../components/ui";
import StripLog, { type Span } from "../viewer/StripLog";
import PageView, { type Fit } from "../viewer/PageView";

type View = "column" | "page";

const VIEWS: { id: View; label: string; title: string }[] = [
  { id: "column", label: "Column", title: "Lithology column" },
  { id: "page", label: "Original scan", title: "Original scan" },
];
const SPANS: { id: Span; label: string }[] = [
  { id: 50, label: "50 ft" },
  { id: 200, label: "200 ft" },
  { id: 1000, label: "1,000 ft" },
  { id: "all", label: "Whole log" },
];
const FITS: { id: Fit; label: string }[] = [
  { id: "width", label: "Fit width" },
  { id: "page", label: "Fit page" },
];

function Segmented<T extends string | number>({ label, items, value, onChange }: { label: string; items: { id: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div role="group" aria-label={label} className="flex rounded-lg bg-panel p-[3px]">
      {items.map((it) => (
        <button
          key={String(it.id)}
          type="button"
          aria-pressed={value === it.id}
          onClick={() => onChange(it.id)}
          className={`h-8 rounded-md px-3 text-xs transition-colors ${value === it.id ? "bg-line-2 text-ink" : "text-muted hover:text-ink"}`}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

export default function ImagePage() {
  const { logId } = useParams();
  const [params, setParams] = useSearchParams();
  const selected = params.get("i") != null ? Number(params.get("i")) : null;
  const { data: log, error } = useLog(logId);
  const { data: intervals } = useIntervals(logId);
  const spanParam = params.get("span");
  const span: Span = spanParam === "all" ? "all" : spanParam === "50" ? 50 : spanParam === "1000" ? 1000 : 200;
  const view: View = params.get("view") === "page" ? "page" : "column";
  const spotlight = params.get("spot") !== "0";
  const fit: Fit = params.get("fit") === "page" ? "page" : "width";
  const setParam = useCallback(
    (key: string, value: string | null) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (value == null) next.delete(key);
          else next.set(key, value);
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );
  const [range, setRange] = useState<[number | null, number | null]>([null, null]);
  const [depthInput, setDepthInput] = useState("");
  const [jump, setJump] = useState<{ depth: number; n: number } | null>(null);

  useEffect(() => {
    if (logId) rememberLog(logId);
    setRange([null, null]);
  }, [logId]);

  const select = useCallback((idx: number) => setParam("i", String(idx)), [setParam]);
  const onView = useCallback((a: number | null, b: number | null) => setRange([a, b]), []);
  const setView = (v: View) => {
    setJump(null); // a fresh view opens where it should, not at an old "go to"
    setParam("view", v === "column" ? null : v);
  };
  const openAt = (depth: number) => {
    setParam("view", null);
    setJump({ depth, n: (jump?.n ?? 0) + 1 });
  };

  if (error) return <div className="p-12"><ErrorNote error={error} /></div>;
  if (!log || !intervals) return null;
  const sel = intervals.find((r) => r.idx === selected) ?? null;
  const ready = log.tiles?.col && log.boxes;
  const title = VIEWS.find((v) => v.id === view)!.title;

  return (
    <div className="mx-auto flex max-w-[1320px] flex-col gap-5 px-6 pb-8 pt-7 lg:px-10">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          <span className="tabular text-[13px] text-muted">
            {view === "page"
              ? "The page as delivered, true to scale · hover for depth, click to open the column there"
              : range[0] != null && range[1] != null
                ? `${fmt(range[0])} – ${fmt(range[1])} ft in view · ${fmt(intervals.length)} beds`
                : `${log.title} · ${fmt(intervals.length)} beds`}
          </span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2.5">
          <Segmented label="View" items={VIEWS} value={view} onChange={setView} />
          {view === "page" ? (
            <>
              <Segmented label="Fit" items={FITS} value={fit} onChange={(f) => setParam("fit", f === "width" ? null : f)} />
              <button
                type="button"
                aria-pressed={spotlight}
                onClick={() => setParam("spot", spotlight ? "0" : null)}
                title="Dim everything except the regions the pipeline reads"
                className={`h-[38px] rounded-lg px-3 text-xs transition-colors ${spotlight ? "bg-line-2 text-ink" : "bg-panel text-muted hover:text-ink"}`}
              >
                Spotlight
              </button>
            </>
          ) : (
            <Segmented label="Feet per screen" items={SPANS} value={span} onChange={(s) => setParam("span", s === 200 ? null : String(s))} />
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const v = Number(depthInput.replace(/[, ]/g, ""));
              if (Number.isFinite(v)) openAt(v);
            }}
            className="flex h-[38px] items-center gap-2 rounded-lg bg-panel px-3 text-xs text-faint"
          >
            <label htmlFor="depth">Go to</label>
            <input
              id="depth"
              inputMode="numeric"
              value={depthInput}
              onChange={(e) => setDepthInput(e.target.value)}
              placeholder={log.top_ft != null ? fmt(Math.round((log.top_ft + (log.bottom_ft ?? log.top_ft)) / 200) * 100) : ""}
              className="w-16 bg-transparent font-mono text-[13px] text-ink outline-none placeholder:text-faint"
            />
            ft
          </form>
        </div>
      </div>

      <div className="flex flex-wrap items-stretch gap-6">
        <section
          aria-label="Scan and predicted beds"
          className="flex h-[calc(100vh-180px)] min-h-[540px] min-w-0 grow-[999] basis-[620px] overflow-hidden rounded-xl border border-line bg-nav"
        >
          {!ready ? (
            <p className="m-auto max-w-sm text-center text-sm text-muted">
              No scan tiles for this log. Run <code className="font-mono">lith dash-build --rebuild-tiles</code> with the dataset present.
            </p>
          ) : view === "page" ? (
            log.tiles?.page ? (
              <PageView key={`page-${log.log_id}`} log={log} intervals={intervals} selected={selected} onSelect={select} fit={fit} spotlight={spotlight} onOpen={openAt} />
            ) : (
              <p className="m-auto max-w-sm text-center text-sm text-muted">
                No whole-page tiles yet. Run <code className="font-mono">lith dash-build --rebuild-tiles</code>.
              </p>
            )
          ) : (
            <StripLog
              key={`col-${log.log_id}`}
              log={log}
              intervals={intervals}
              selected={selected}
              onSelect={select}
              span={span}
              jump={jump}
              onView={onView}
            />
          )}
        </section>

        <aside className={`${view === "page" ? "hidden" : "flex"} h-[calc(100vh-180px)] min-h-[540px] min-w-0 grow basis-[280px] flex-col gap-5 lg:max-w-[340px]`}>
          <SelectedBed log={log} iv={sel} count={intervals.length} onStep={(d) => sel && select(Math.min(intervals.length - 1, Math.max(0, sel.idx + d)))} />
          <BedList intervals={intervals} selected={selected} range={range} onSelect={select} />
          <LegendMatch log={log} />
        </aside>
      </div>
    </div>
  );
}

function SelectedBed({ log, iv, count, onStep }: { log: LogDetail; iv: Interval | null; count: number; onStep: (d: number) => void }) {
  return (
    <section aria-label="Selected bed" className="flex flex-col gap-3 rounded-xl border border-line p-4">
      <div className="flex items-center">
        <span className="text-xs text-faint">{iv ? `Bed #${iv.idx + 1} of ${count}` : "Selected bed"}</span>
        {iv && (
          <div className="ml-auto flex gap-1.5">
            <button type="button" aria-label="Previous bed" disabled={iv.idx === 0} onClick={() => onStep(-1)} className="flex h-8 w-8 items-center justify-center rounded-md bg-panel text-muted hover:text-ink disabled:opacity-40">
              <ChevronUp size={14} aria-hidden="true" />
            </button>
            <button type="button" aria-label="Next bed" disabled={iv.idx === count - 1} onClick={() => onStep(1)} className="flex h-8 w-8 items-center justify-center rounded-md bg-panel text-muted hover:text-ink disabled:opacity-40">
              <ChevronDown size={14} aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
      {!iv ? (
        <p className="m-0 text-[13px] leading-relaxed text-muted">Click a bed, a label or a row below.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <Dot color={iv.color} unresolved={iv.unresolved} similar={iv.lookalike} size={10} />
            <span className={`text-base font-semibold ${iv.unresolved ? "text-rose" : ""}`}>{bedName(iv)}</span>
            {iv.cluster_id != null && <span className="text-[13px] text-muted">cluster {iv.cluster_id}</span>}
          </div>
          {iv.lookalike && (
            <p className="m-0 text-[13px] leading-relaxed text-muted">
              Similar types: computer vision narrowed this bed to these look-alike legend entries but cannot pick one.
              The CSV holds its best guess, {iv.rock}.
            </p>
          )}
          {iv.unresolved && iv.notes && <p className="m-0 text-[13px] leading-relaxed text-muted">{iv.notes}</p>}
          <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
            <dt className="text-faint">Depth</dt>
            <dd className="tabular m-0 font-mono">
              {fmt(iv.top_depth, 2)} – {fmt(iv.bottom_depth, 2)} ft
            </dd>
            <dt className="text-faint">Thickness</dt>
            <dd className="tabular m-0 font-mono">{fmt(iv.thickness_ft, 1)} ft</dd>
            <dt className="text-faint">Pixels</dt>
            <dd className="tabular m-0 font-mono">
              y {iv.top_y} – {iv.bottom_y}
            </dd>
            <dt className="text-faint">Confidence</dt>
            <dd className="tabular m-0 font-mono">{fmt(iv.confidence, 2)}</dd>
          </dl>
          <Link to={`/csv/${log.log_id}?i=${iv.idx}`} className="text-[13px] text-muted no-underline hover:text-ink">
            View row in CSV →
          </Link>
        </>
      )}
    </section>
  );
}

function BedList({
  intervals,
  selected,
  range,
  onSelect,
}: {
  intervals: Interval[];
  selected: number | null;
  range: [number | null, number | null];
  onSelect: (idx: number) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [userScrolling, setUserScrolling] = useState(false);
  const virt = useVirtualizer({ count: intervals.length, getScrollElement: () => scroller.current, estimateSize: () => 34, overscan: 10 });
  const inView = (r: Interval) =>
    range[0] != null && range[1] != null && r.bottom_depth != null && r.top_depth != null && r.bottom_depth > range[0] && r.top_depth < range[1];
  const first = useMemo(() => intervals.findIndex(inView), [intervals, range[0], range[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!userScrolling && first >= 0) virt.scrollToIndex(first, { align: "start" });
  }, [first]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const at = intervals.findIndex((r) => r.idx === selected);
    if (at >= 0 && !userScrolling) virt.scrollToIndex(at, { align: "auto" });
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section aria-label="Beds" className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-line">
      <div className="flex items-center border-b border-line px-4 py-2.5 text-xs text-faint">
        Beds
        <span className="ml-auto">in view are bright</span>
      </div>
      <div
        ref={scroller}
        onMouseEnter={() => setUserScrolling(true)}
        onMouseLeave={() => setUserScrolling(false)}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        <div className="relative" style={{ height: virt.getTotalSize() }}>
          {virt.getVirtualItems().map((v) => {
            const r = intervals[v.index];
            const on = inView(r);
            return (
              <button
                key={v.key}
                type="button"
                onClick={() => onSelect(r.idx)}
                style={{ transform: `translateY(${v.start}px)` }}
                className={`absolute left-0 right-0 top-0 flex h-[34px] items-center gap-2.5 px-4 text-left text-[13px] transition-colors hover:bg-hover ${
                  r.idx === selected ? "bg-panel-2" : ""
                }`}
              >
                <Dot color={r.color} unresolved={r.unresolved} similar={r.lookalike} />
                <span className="w-9 shrink-0 font-mono text-[11px] text-faint">#{r.idx + 1}</span>
                <span className={`truncate ${on ? (r.unresolved ? "text-rose" : "text-ink") : "text-faint"}`}>{bedName(r)}</span>
                <span className={`tabular ml-auto shrink-0 font-mono text-[11px] ${on ? "text-muted" : "text-faint/70"}`}>{fmt(r.top_depth, 1)}</span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function LegendMatch({ log }: { log: LogDetail }) {
  const used = log.legend.filter((e) => e.used);
  const unused = log.legend.filter((e) => !e.used && e.is_rock).length;
  const symbols = log.legend.filter((e) => !e.is_rock).length;
  return (
    <details className="group rounded-xl border border-line px-4 py-3 text-xs text-faint">
      <summary className="cursor-pointer select-none hover:text-ink">Legend match ({used.length} used)</summary>
      <div className="mt-3 flex flex-col gap-2.5">
        {used.map((e) => (
          <div key={e.idx} className="flex items-center gap-2.5 text-[13px] text-ink">
            {e.swatch_url ? (
              <img src={e.swatch_url} alt="" className="h-3.5 w-[22px] shrink-0 border border-line-2 object-cover" />
            ) : (
              <Dot color={e.color ?? "#7E8794"} size={12} />
            )}
            <span className="truncate">{e.label}</span>
            <span className="tabular ml-auto shrink-0 font-mono text-xs text-muted">
              {e.cluster_id != null ? `c${e.cluster_id} · ` : ""}
              {e.beds}
            </span>
          </div>
        ))}
        <p className="m-0 leading-relaxed">
          {unused ? `${unused} more legend ${unused === 1 ? "entry" : "entries"} not in this log` : ""}
          {unused && symbols ? " · " : ""}
          {symbols ? `${symbols} drilling ${symbols === 1 ? "symbol" : "symbols"} ignored` : ""}
        </p>
        {log.tiles?.legend && <img src={`/tiles/${log.log_id}/legend.webp`} alt="The log's legend" className="w-full rounded-sm bg-paper" />}
      </div>
    </details>
  );
}
