import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Upload } from "lucide-react";
import {
  layoutCropUrl,
  startDetect,
  useDetect,
  useLayout,
  useLayoutPage,
  type BoxKey,
  type LayoutDetail,
  type LayoutRow,
} from "../api";
import { ErrorNote, Spinner, Stat } from "../components/ui";

const BOXES: { key: BoxKey; label: string; short: string; color: string }[] = [
  { key: "depth_column", label: "Depth column", short: "Depth", color: "#4FA3F7" },
  { key: "interpreted_lithology_column", label: "Lithology column", short: "Lith", color: "#B79CFF" },
  { key: "lithology_legend", label: "Legend", short: "Legend", color: "#F2B544" },
];

const GOOD = "#7CC9A0";
const OK = "#E8B04B";
const BAD = "#E39AB0";

function tone(v: number | null | undefined): string {
  if (v == null) return "var(--color-faint)";
  return v >= 0.8 ? GOOD : v >= 0.5 ? OK : BAD;
}

const pct = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));

/** Doc ids start with long API numbers: show the readable tail. */
function shortTitle(id: string): string {
  return id.replace(/^(\d{6,}_)+/, "").replace(/_/g, " ");
}

function minIou(r: LayoutRow): number {
  if (!r.iou) return 1;
  return Math.min(...BOXES.map((b) => r.iou?.[b.key] ?? 1));
}

function Pill({ v, label }: { v: number | null | undefined; label: string }) {
  return (
    <span
      className="tabular inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium"
      style={{ color: tone(v), backgroundColor: "rgba(255,255,255,0.04)" }}
      title={`${label} IoU vs annotation`}
    >
      {pct(v)}
    </span>
  );
}

// ---------------------------------------------------------------------------------------------------------- page

export default function LayoutPage() {
  const { docId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { data, error, isLoading } = useLayout();
  const [filter, setFilter] = useState<"all" | "attention">("all");
  const [vendor, setVendor] = useState("all");
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState<{ name: string; pct: number } | null>(null);
  const [drag, setDrag] = useState(false);
  const job = useDetect(jobId);
  const busy = !!uploading || !!jobId;

  const rows = useMemo(() => {
    const all = [...(data?.pages ?? [])].sort((a, b) => minIou(a) - minIou(b));
    return all.filter(
      (r) =>
        (vendor === "all" || r.vendor === vendor) &&
        (filter === "all" || minIou(r) < 0.8 || BOXES.some((b) => (r.confidence?.[b.key] ?? 1) < 0.5)),
    );
  }, [data, filter, vendor]);
  const vendors = useMemo(() => [...new Set((data?.pages ?? []).map((r) => r.vendor))].sort(), [data]);

  const selected = docId ?? rows[0]?.doc_id;

  useEffect(() => {
    if (!jobId) return;
    if (job.data?.state === "done") {
      qc.invalidateQueries({ queryKey: ["layout"] });
      navigate(`/layout/${encodeURIComponent(jobId)}`);
      setJobId(null);
    } else if (job.data?.state === "error") {
      setUploadErr(`Detection failed: ${job.data.error}`);
      setJobId(null); // free the button so another image can be tried
    } else if (job.error) {
      setUploadErr(job.error instanceof Error ? job.error.message : String(job.error));
      setJobId(null);
    }
  }, [job.data?.state, job.data?.error, job.error, jobId, navigate, qc]);

  async function onFile(f: File | undefined) {
    if (!f || busy) return;
    setUploadErr(null);
    setUploading({ name: f.name, pct: 0 });
    try {
      setJobId(await startDetect(f, (pct) => setUploading({ name: f.name, pct })));
    } catch (e) {
      setUploadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(null);
    }
  }

  if (isLoading) return <p className="p-12 text-sm text-muted">Loading…</p>;
  if (error) return <div className="p-12"><ErrorNote error={error} /></div>;
  if (!data?.run)
    return (
      <p className="p-12 text-sm text-muted">
        No layout run yet. Run <code className="font-mono">lith layout dataset/images --out out/layout</code>.
      </p>
    );

  const dev = data.metrics?.dev;

  return (
    <div className="flex flex-col gap-8 px-6 py-8 lg:px-12">
      <section
        className={`-m-3 flex flex-col gap-2 rounded-xl p-3 transition-colors ${drag ? "bg-hover outline-2 outline-dashed outline-select" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          onFile(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="m-0 text-lg font-semibold tracking-tight">Layout detector</h1>
            <p className="m-0 mt-1 text-sm text-muted">
              Step 1 of the pipeline: the depth column, lithology column and legend, found without the annotation files.
              Scores are overlap (IoU) with the annotated boxes.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1">
            {/* a real <label> for the file input: opens the picker natively in every browser */}
            <label
              aria-disabled={busy}
              className={`flex h-10 cursor-pointer items-center gap-2 rounded-lg bg-accent px-3.5 text-sm font-medium text-white transition-colors focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-select hover:bg-accent-hi ${
                busy ? "pointer-events-none opacity-60" : ""
              }`}
            >
              <Upload size={15} strokeWidth={1.75} aria-hidden="true" />
              Detect on an image
              <input
                type="file"
                accept=".tif,.tiff,.TIF,.TIFF,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg"
                className="sr-only"
                disabled={busy}
                onChange={(e) => {
                  onFile(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
            <span className="text-[11px] text-faint">or drop a TIFF / PNG / JPG here</span>
          </div>
        </div>
        <div role="status" aria-live="polite" className="min-h-5 text-sm">
          {uploading && (
            <span className="flex items-center gap-2 text-muted">
              <Spinner /> Uploading {uploading.name}… {uploading.pct}%
            </span>
          )}
          {!uploading && jobId && (
            <span className="flex items-center gap-2 text-muted">
              <Spinner /> Detecting boxes… {job.data ? `${job.data.elapsed.toFixed(0)} s` : ""}
            </span>
          )}
          {!busy && uploadErr && <span className="text-rose">{uploadErr}</span>}
        </div>
      </section>

      {dev && (
        <section aria-label="Scores" className="grid grid-cols-2 gap-6 rounded-xl border border-line bg-panel p-5 md:grid-cols-4">
          {BOXES.map((b) => (
            <div key={b.key} className="flex flex-col gap-1.5">
              <span className="flex items-center gap-2 text-xs text-faint">
                <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: b.color }} />
                {b.label}
              </span>
              <span className="tabular text-2xl font-medium tracking-tight" style={{ color: tone(dev[b.key].vendor_macro) }}>
                {dev[b.key].vendor_macro.toFixed(3)}
              </span>
              <span className="text-xs text-muted">
                {dev[b.key]["iou_gt_0.5"]}/{dev.pages} pages above 0.5
              </span>
            </div>
          ))}
          <Stat
            label="Speed"
            value={data.mean_seconds != null ? `${data.mean_seconds} s/page` : "–"}
            hint={`slowest page ${data.max_seconds} s · run ${data.run}`}
          />
        </section>
      )}

      <section className="grid min-w-0 gap-6 lg:grid-cols-[minmax(280px,360px)_1fr]">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {(["all", "attention"] as const).map((f) => (
              <button
                key={f}
                type="button"
                aria-pressed={filter === f}
                onClick={() => setFilter(f)}
                className={`h-8 rounded-md px-2.5 text-xs transition-colors ${
                  filter === f ? "bg-panel-2 text-ink" : "text-muted hover:bg-hover"
                }`}
              >
                {f === "all" ? `All pages (${data.pages.length})` : "Needs a look"}
              </button>
            ))}
            <label className="ml-auto flex items-center gap-1.5 text-xs text-faint">
              Vendor
              <select
                value={vendor}
                onChange={(e) => setVendor(e.target.value)}
                className="h-8 rounded-md border border-line-2 bg-panel px-1.5 text-xs text-ink"
              >
                <option value="all">All</option>
                {vendors.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="m-0 text-xs text-faint">Worst first. Columns: depth · lithology · legend IoU.</p>
          <ul className="m-0 flex max-h-[70vh] list-none flex-col gap-0.5 overflow-y-auto p-0 pr-1">
            {(data.uploads ?? [])
              .filter((u) => u.state === "done")
              .map((u) => (
                <li key={u.id}>
                  <RowButton
                    active={selected === u.id}
                    onClick={() => navigate(`/layout/${encodeURIComponent(u.id)}`)}
                    title={u.name}
                    sub="your image · no annotation"
                  />
                </li>
              ))}
            {rows.map((r) => (
              <li key={r.doc_id}>
                <RowButton
                  active={selected === r.doc_id}
                  onClick={() => navigate(`/layout/${encodeURIComponent(r.doc_id)}`)}
                  title={shortTitle(r.doc_id)}
                  sub={r.vendor}
                  pills={BOXES.map((b) => (
                    <Pill key={b.key} v={r.iou?.[b.key]} label={b.label} />
                  ))}
                />
              </li>
            ))}
            {rows.length === 0 && <li className="px-3 py-2 text-sm text-muted">Nothing matches the filter.</li>}
          </ul>
        </div>
        {selected ? <PageViewer id={selected} /> : null}
      </section>
    </div>
  );
}

function RowButton({
  active,
  onClick,
  title,
  sub,
  pills,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  sub: string;
  pills?: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "true" : undefined}
      className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors ${
        active ? "bg-panel-2" : "hover:bg-hover"
      }`}
    >
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm text-ink" title={title}>
          {title}
        </span>
        <span className="text-[11px] text-faint">{sub}</span>
      </span>
      {pills && <span className="flex shrink-0 gap-0.5">{pills}</span>}
    </button>
  );
}

// ---------------------------------------------------------------------------------------------------------- viewer

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(900);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return { ref, w };
}

function PageViewer({ id }: { id: string }) {
  const { data, error, isLoading } = useLayoutPage(id);
  if (isLoading) return <p className="text-sm text-muted">Loading page…</p>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;
  return <Viewer key={id} d={data} />;
}

function Viewer({ d }: { d: LayoutDetail }) {
  const [showDet, setShowDet] = useState(true);
  const [showAnn, setShowAnn] = useState(true);
  const [view, setView] = useState<"header" | "body">("header");
  const [pos, setPos] = useState(0.5);
  const [committed, setCommitted] = useState(0.5);
  const { ref, w } = useWidth<HTMLDivElement>();
  const [loaded, setLoaded] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setCommitted(pos), 250);
    return () => clearTimeout(t);
  }, [pos]);

  if (!d.size) return <p className="text-sm text-muted">Image not available.</p>;
  const [W, H] = d.size;
  const dpi = d.dpi ?? W / 8.5;
  const [top, bot] = d.body ?? [0, H];
  let y0: number;
  let y1: number;
  if (view === "header") {
    y0 = 0;
    y1 = Math.min(H, top + 0.75 * dpi);
  } else {
    const band = Math.min(bot - top, 3 * dpi);
    y0 = top + committed * Math.max(0, bot - top - band);
    y1 = y0 + band;
  }
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  const url = layoutCropUrl(d.doc_id, y0, y1, Math.min(1800, w * dpr));
  const bodyFt = d.body ? ((bot - top) / dpi).toFixed(0) : null;

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-col gap-1">
        <h2 className="m-0 truncate text-base font-semibold tracking-tight" title={d.doc_id}>
          {d.name ?? shortTitle(d.doc_id)}
        </h2>
        <p className="m-0 text-xs text-faint">
          {d.source === "upload" ? "Your image (no annotation to compare with)" : `${d.vendor} · ${d.split}`} · {W}×{H} px ·{" "}
          {Math.round(dpi)} dpi{d.seconds != null ? ` · detected in ${d.seconds} s` : ""}
          {bodyFt ? ` · log body ${bodyFt} in tall` : ""}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {BOXES.map((b) => {
          const iou = d.iou?.[b.key];
          const conf = d.confidence?.[b.key];
          const det = d.boxes[b.key];
          const ann = d.annotated?.[b.key];
          return (
            <div key={b.key} className="flex flex-col gap-1.5 rounded-lg border border-line bg-panel p-3">
              <span className="flex items-center gap-2 text-xs text-muted">
                <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: b.color }} />
                {b.label}
              </span>
              <div className="flex items-baseline gap-3">
                {d.annotated && (
                  <span className="tabular text-xl font-medium" style={{ color: tone(iou) }}>
                    {pct(iou)}
                    <span className="ml-1 text-[11px] font-normal text-faint">IoU</span>
                  </span>
                )}
                <span className="tabular text-sm" style={{ color: conf != null && conf < 0.5 ? BAD : "var(--color-muted)" }}>
                  conf {pct(conf)}
                </span>
              </div>
              <span className="tabular font-mono text-[11px] leading-relaxed text-faint">
                detected x {det[0]}–{det[2]}
                {ann ? (
                  <>
                    <br />
                    annotated x {ann[0]}–{ann[2]}
                  </>
                ) : null}
              </span>
              {conf != null && conf < 0.5 && <span className="text-[11px] text-rose">Low confidence: check by eye</span>}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {(["header", "body"] as const).map((v) => (
          <button
            key={v}
            type="button"
            aria-pressed={view === v}
            onClick={() => setView(v)}
            className={`h-8 rounded-md px-2.5 text-xs transition-colors ${view === v ? "bg-panel-2 text-ink" : "text-muted hover:bg-hover"}`}
          >
            {v === "header" ? "Header & legend" : "Log body"}
          </button>
        ))}
        <span className="mx-1 h-5 w-px bg-line" aria-hidden="true" />
        <Toggle on={showDet} set={setShowDet} label="Detected" swatch={<span className="inline-block h-2.5 w-3 rounded-sm border-2 border-select" />} />
        {d.annotated && (
          <Toggle
            on={showAnn}
            set={setShowAnn}
            label="Annotated"
            swatch={<span className="inline-block h-2.5 w-3 rounded-sm border border-dashed border-ink" />}
          />
        )}
      </div>

      {view === "body" && (
        <label className="flex items-center gap-3 text-xs text-faint">
          <span>Top</span>
          <input
            type="range"
            min={0}
            max={1000}
            value={Math.round(pos * 1000)}
            onChange={(e) => setPos(Number(e.target.value) / 1000)}
            className="flex-1 accent-[#8f74ea]"
            aria-label="Position in the log body"
          />
          <span>Bottom</span>
        </label>
      )}

      <div ref={ref} className="max-h-[78vh] overflow-auto rounded-lg border border-line bg-paper">
        <div className="relative w-full">
          {loaded !== url && (
            <div className="absolute inset-x-0 top-0 flex items-center gap-2 p-3 text-xs text-[#555]">
              <Spinner /> Rendering…
            </div>
          )}
          <img
            src={url}
            alt={`${view === "header" ? "Header" : "Body section"} of ${d.doc_id} with the boxes drawn on`}
            className="block h-auto w-full"
            style={{ aspectRatio: `${W} / ${Math.max(1, y1 - y0)}` }}
            onLoad={() => setLoaded(url)}
          />
          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox={`0 ${y0} ${W} ${Math.max(1, y1 - y0)}`}
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {BOXES.map((b) => {
              const det = d.boxes[b.key];
              const ann = d.annotated?.[b.key];
              return (
                <g key={b.key}>
                  {showAnn && ann && (
                    <>
                      <rect x={ann[0]} y={ann[1]} width={ann[2] - ann[0]} height={ann[3] - ann[1]} fill="none" stroke="#fff" strokeWidth={4} vectorEffect="non-scaling-stroke" opacity={0.85} />
                      <rect x={ann[0]} y={ann[1]} width={ann[2] - ann[0]} height={ann[3] - ann[1]} fill="none" stroke="#111" strokeWidth={1.5} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />
                    </>
                  )}
                  {showDet && (
                    <rect
                      x={det[0]}
                      y={det[1]}
                      width={det[2] - det[0]}
                      height={det[3] - det[1]}
                      fill={b.color}
                      fillOpacity={0.12}
                      stroke={b.color}
                      strokeWidth={2.5}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {d.notes.length > 0 && (
        <ul className="m-0 flex list-disc flex-col gap-1 pl-5 text-xs text-muted">
          {d.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Toggle({ on, set, label, swatch }: { on: boolean; set: (v: boolean) => void; label: string; swatch: ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => set(!on)}
      className={`flex h-8 items-center gap-2 rounded-md px-2.5 text-xs transition-colors ${on ? "bg-panel-2 text-ink" : "text-faint hover:bg-hover"}`}
    >
      {swatch}
      {label}
    </button>
  );
}
