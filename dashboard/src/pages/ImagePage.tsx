import { useCallback, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useIntervals, useLog, type Interval, type LogDetail } from "../api";
import { fmt, rememberLog } from "../format";
import { Dot, ErrorNote } from "../components/ui";
import StripViewer, { type Zoom } from "../viewer/StripViewer";

const ZOOMS: { id: Zoom; label: string }[] = [
  { id: "fit", label: "Fit" },
  { id: "100", label: "100%" },
  { id: "200", label: "200%" },
];

export default function ImagePage() {
  const { logId } = useParams();
  const [params, setParams] = useSearchParams();
  const selected = params.get("i") != null ? Number(params.get("i")) : null;
  const { data: log, error } = useLog(logId);
  const { data: intervals } = useIntervals(logId);
  const [zoom, setZoom] = useState<Zoom>("fit");
  const [range, setRange] = useState<[number | null, number | null]>([null, null]);
  const [depthInput, setDepthInput] = useState("");
  const [jump, setJump] = useState<{ depth: number; n: number } | null>(null);

  useEffect(() => {
    if (logId) rememberLog(logId);
  }, [logId]);

  const select = useCallback((idx: number) => setParams({ i: String(idx) }, { replace: true }), [setParams]);
  const onView = useCallback((a: number | null, b: number | null) => setRange([a, b]), []);

  if (error) return <div className="p-12"><ErrorNote error={error} /></div>;
  if (!log || !intervals) return null;
  const sel = intervals.find((r) => r.idx === selected) ?? null;

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-6 pb-10 pt-8 lg:px-12">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">Scan overlay</h1>
          <span className="tabular text-[13px] text-muted">
            {range[0] != null && range[1] != null ? `Showing ${fmt(range[0])} – ${fmt(range[1])} ft` : log.title}
          </span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2.5">
          <div role="group" aria-label="Zoom" className="flex rounded-lg bg-panel p-[3px]">
            {ZOOMS.map((z) => (
              <button
                key={z.id}
                type="button"
                aria-pressed={zoom === z.id}
                onClick={() => setZoom(z.id)}
                className={`h-8 rounded-md px-3 text-xs transition-colors ${zoom === z.id ? "bg-line-2 text-ink" : "text-muted hover:text-ink"}`}
              >
                {z.label}
              </button>
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const v = Number(depthInput.replace(/[, ]/g, ""));
              if (Number.isFinite(v)) setJump({ depth: v, n: (jump?.n ?? 0) + 1 });
            }}
            className="flex h-[38px] items-center gap-2 rounded-lg bg-panel px-3 text-xs text-faint"
          >
            <label htmlFor="depth">Depth</label>
            <input
              id="depth"
              inputMode="numeric"
              value={depthInput}
              onChange={(e) => setDepthInput(e.target.value)}
              placeholder={log.top_ft != null ? fmt(Math.round(((log.top_ft + (log.bottom_ft ?? log.top_ft)) / 2) / 100) * 100) : ""}
              className="w-16 bg-transparent font-mono text-[13px] text-ink outline-none placeholder:text-faint"
            />
            ft
          </form>
        </div>
      </div>

      <div className="flex flex-wrap items-start gap-8">
        <section
          aria-label="Scan with predicted beds"
          className="flex h-[calc(100vh-205px)] min-h-[520px] min-w-0 grow-[999] basis-[600px] overflow-hidden rounded-xl border border-line bg-nav"
        >
          {log.tiles && log.boxes ? (
            <StripViewer log={log} intervals={intervals} selected={selected} onSelect={select} zoom={zoom} jump={jump} onView={onView} />
          ) : (
            <p className="m-auto max-w-sm text-center text-sm text-muted">
              No scan tiles for this log. Run <code className="font-mono">lith dash-build</code> with the dataset present.
            </p>
          )}
        </section>

        <aside className="flex min-w-0 grow basis-[260px] flex-col gap-7">
          <SelectedBed log={log} iv={sel} count={intervals.length} onStep={(d) => sel && select(Math.min(intervals.length - 1, Math.max(0, sel.idx + d)))} />
          <div className="h-px bg-line" />
          <LegendMatch log={log} />
        </aside>
      </div>
    </div>
  );
}

function SelectedBed({ log, iv, count, onStep }: { log: LogDetail; iv: Interval | null; count: number; onStep: (d: number) => void }) {
  return (
    <section aria-label="Selected bed" className="flex flex-col gap-4">
      <div className="flex items-center">
        <span className="text-xs text-faint">Selected bed</span>
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
        <p className="text-sm leading-relaxed text-muted">Click a bed on the scan, or a row in the CSV view, to see its details.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <Dot color={iv.color} unresolved={iv.unresolved} size={10} />
            <span className={`text-lg font-semibold ${iv.unresolved ? "text-rose" : ""}`}>{iv.rock}</span>
            {iv.cluster_id != null && <span className="text-[13px] text-muted">cluster {iv.cluster_id}</span>}
          </div>
          {iv.unresolved && iv.notes && <p className="text-[13px] leading-relaxed text-muted">{iv.notes}</p>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2.5 text-[13px]">
            <dt className="text-faint">Depth</dt>
            <dd className="tabular m-0 font-mono">
              {fmt(iv.top_depth, 2)} – {fmt(iv.bottom_depth, 2)} ft
            </dd>
            <dt className="text-faint">Thickness</dt>
            <dd className="tabular m-0 font-mono">{fmt(iv.thickness_ft, 1)} ft</dd>
            <dt className="text-faint">Pixels</dt>
            <dd className="tabular m-0 font-mono">
              y {fmt(iv.top_y)} – {fmt(iv.bottom_y)}
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

function LegendMatch({ log }: { log: LogDetail }) {
  const used = log.legend.filter((e) => e.used);
  const unused = log.legend.filter((e) => !e.used && e.is_rock).length;
  const symbols = log.legend.filter((e) => !e.is_rock).length;
  return (
    <section aria-label="Legend match" className="flex flex-col gap-3.5">
      <span className="text-xs text-faint">Legend match</span>
      {used.map((e) => (
        <div key={e.idx} className="flex items-center gap-2.5 text-[13px]">
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
      {!used.length && <p className="text-[13px] text-muted">No legend entry matched with confidence.</p>}
      <p className="text-xs leading-relaxed text-faint">
        {unused ? `${unused} more legend ${unused === 1 ? "entry" : "entries"} not found in this log` : ""}
        {unused && symbols ? " · " : ""}
        {symbols ? `${symbols} drilling ${symbols === 1 ? "symbol" : "symbols"} ignored` : ""}
      </p>
      {log.tiles?.legend && (
        <details className="text-xs text-faint">
          <summary className="cursor-pointer select-none hover:text-ink">Legend image</summary>
          <img src={`/tiles/${log.log_id}/legend.webp`} alt="The log's legend" className="mt-2.5 w-full rounded-sm bg-paper" />
        </details>
      )}
    </section>
  );
}
