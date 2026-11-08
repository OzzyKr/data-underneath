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
import { fmt } from "../format";
import { ErrorNote, Segmented, Spinner } from "../components/ui";

const BOXES: { key: BoxKey; label: string; short: string; color: string }[] = [
  { key: "depth_column", label: "Depth column", short: "Depth", color: "#4FA3F7" },
  { key: "interpreted_lithology_column", label: "Lithology column", short: "Lith.", color: "#B79CFF" },
  { key: "lithology_legend", label: "Legend", short: "Legend", color: "#F2B544" },
];

const FLAG_IOU = 0.8; // a page is flagged when a box overlaps the annotation less than this...
const LOW_CONF = 0.5; // ...or the detector's own confidence is below this

const num = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));
const lowClass = (v: number | null | undefined, below: number) => (v != null && v < below ? "text-rose" : "");

/** Doc ids start with long API numbers: show the readable tail. */
function shortTitle(id: string): string {
  return id.replace(/^(\d{6,}_)+/, "").replace(/_/g, " ");
}

function minIou(r: LayoutRow): number {
  if (!r.iou) return 1;
  return Math.min(...BOXES.map((b) => r.iou?.[b.key] ?? 1));
}

const flagged = (r: LayoutRow) => minIou(r) < FLAG_IOU || BOXES.some((b) => (r.confidence?.[b.key] ?? 1) < LOW_CONF);

// ---------------------------------------------------------------------------------------------------------- page

export default function LayoutPage() {
  const { docId } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { data, error, isLoading } = useLayout();
  const [filter, setFilter] = useState<"all" | "flagged">("all");
  const [vendor, setVendor] = useState("all");
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState<{ name: string; pct: number } | null>(null);
  const [drag, setDrag] = useState(false);
  const job = useDetect(jobId);
  const busy = !!uploading || !!jobId;

  const pages = useMemo(() => [...(data?.pages ?? [])].sort((a, b) => minIou(a) - minIou(b)), [data]);
  const nFlagged = useMemo(() => pages.filter(flagged).length, [pages]);
  const rows = useMemo(
    () => pages.filter((r) => (vendor === "all" || r.vendor === vendor) && (filter === "all" || flagged(r))),
    [pages, filter, vendor],
  );
  const vendors = useMemo(() => [...new Set(pages.map((r) => r.vendor))].sort(), [pages]);
  const dupTitles = useMemo(() => {
    const seen = new Map<string, number>();
    pages.forEach((r) => seen.set(r.title ?? r.doc_id, (seen.get(r.title ?? r.doc_id) ?? 0) + 1));
    return new Set([...seen].filter(([, n]) => n > 1).map(([t]) => t));
  }, [pages]);
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
        No layout run. <code className="font-mono">lith layout dataset/images --out out/layout</code>
      </p>
    );

  const dev = data.metrics?.dev;

  return (
    <div
      className={`flex flex-col gap-8 px-6 py-8 transition-colors lg:px-12 ${drag ? "bg-hover" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={(e) => e.currentTarget === e.target && setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        onFile(e.dataTransfer.files?.[0]);
      }}
    >
      <h1 className="sr-only">Layout detection</h1>

      <section className="flex flex-wrap items-end justify-between gap-6">
        {dev && (
          <div className="flex flex-col gap-3">
            <dl className="m-0 flex flex-wrap gap-x-12 gap-y-4">
              {BOXES.map((b) => (
                <Metric key={b.key} label={b.label} value={dev[b.key].vendor_macro.toFixed(3)} />
              ))}
              <Metric label="Time per page" value={data.mean_seconds != null ? `${fmt(data.mean_seconds, 1)} s` : "–"} />
            </dl>
            <span className="text-xs text-faint">Mean IoU vs annotation, {dev.pages} dev pages</span>
          </div>
        )}
        <div className="flex items-center gap-4">
          <span role="status" aria-live="polite" className="text-xs text-muted">
            {uploading ? (
              <span className="flex items-center gap-2">
                <Spinner /> {uploading.pct}%
              </span>
            ) : jobId ? (
              <span className="flex items-center gap-2">
                <Spinner /> {job.data ? `${job.data.elapsed.toFixed(0)} s` : ""}
              </span>
            ) : uploadErr ? (
              <span className="text-rose">{uploadErr}</span>
            ) : null}
          </span>
          {/* a real <label> for the file input: opens the picker natively in every browser */}
          <label
            aria-disabled={busy}
            className={`flex h-10 cursor-pointer items-center gap-2 rounded-lg border border-line-2 px-3.5 text-sm text-ink transition-colors focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-select hover:bg-hover ${
              busy ? "pointer-events-none opacity-60" : ""
            }`}
          >
            <Upload size={15} strokeWidth={1.75} aria-hidden="true" />
            Upload scan
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
        </div>
      </section>

      <section className="grid min-w-0 gap-8 lg:grid-cols-[minmax(300px,380px)_1fr]">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Segmented
              label="Pages"
              items={[
                { id: "all", label: `All ${pages.length}` },
                { id: "flagged", label: `Flagged ${nFlagged}` },
              ]}
              value={filter}
              onChange={setFilter}
            />
            <select
              value={vendor}
              onChange={(e) => setVendor(e.target.value)}
              aria-label="Vendor"
              className="ml-auto h-[38px] rounded-lg border border-line-2 bg-panel px-2 text-xs text-ink"
            >
              <option value="all">All vendors</option>
              {vendors.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          <div className="flex min-w-0 flex-col">
            <div className="grid grid-cols-[1fr_48px_48px_52px] gap-2 border-b border-line px-3 pb-2 text-[11px] text-faint">
              <span>Page</span>
              {BOXES.map((b) => (
                <span key={b.key} className="text-right">
                  {b.short}
                </span>
              ))}
            </div>
            <ul className="m-0 flex max-h-[70vh] list-none flex-col overflow-y-auto p-0">
              {(data.uploads ?? [])
                .filter((u) => u.state === "done")
                .map((u) => (
                  <li key={u.id}>
                    <Row active={selected === u.id} onClick={() => navigate(`/layout/${encodeURIComponent(u.id)}`)} title={u.name} />
                  </li>
                ))}
              {rows.map((r) => (
                <li key={r.doc_id}>
                  <Row
                    active={selected === r.doc_id}
                    onClick={() => navigate(`/layout/${encodeURIComponent(r.doc_id)}`)}
                    title={r.title ?? shortTitle(r.doc_id)}
                    variant={dupTitles.has(r.title ?? r.doc_id) ? r.variant : undefined}
                    full={r.doc_id}
                    values={BOXES.map((b) => r.iou?.[b.key])}
                  />
                </li>
              ))}
              {rows.length === 0 && <li className="px-3 py-2 text-sm text-muted">No pages.</li>}
            </ul>
          </div>
        </div>
        {selected ? <PageViewer id={selected} /> : null}
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs text-faint">{label}</dt>
      <dd className="m-0 tabular text-2xl font-medium tracking-tight text-ink">{value}</dd>
    </div>
  );
}

function Row({
  active,
  onClick,
  title,
  variant,
  full,
  values,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  variant?: string;
  full?: string;
  values?: (number | null | undefined)[];
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "true" : undefined}
      className={`grid w-full grid-cols-[1fr_48px_48px_52px] items-center gap-2 border-b border-line/60 px-3 py-2 text-left transition-colors ${
        active ? "bg-panel-2" : "hover:bg-hover"
      }`}
    >
      <span className="truncate text-sm text-ink" title={full ?? title}>
        {title}
        {variant && <span className="ml-2 text-xs text-faint">{variant}</span>}
      </span>
      {(values ?? [null, null, null]).map((v, i) => (
        <span key={i} className={`tabular text-right text-xs text-muted ${lowClass(v, 0.5)}`}>
          {num(v)}
        </span>
      ))}
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
  if (isLoading) return <p className="text-sm text-muted">Loading…</p>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;
  return <Viewer key={id} d={data} />;
}

// Annotated boxes: a thick black dash on a white halo, drawn over the detected boxes so they read on ink and paper.
const ANN_HALO = 8;
const ANN_STROKE = 3;
const ANN_DASH = "12 7";

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
  const meta = [d.source === "upload" ? "Upload" : d.vendor, `${fmt(W)} × ${fmt(H)} px`, `${Math.round(dpi)} dpi`, d.seconds != null ? `${d.seconds} s` : null]
    .filter(Boolean)
    .join(", ");

  return (
    <div className="flex min-w-0 flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2 className="m-0 truncate text-base font-semibold tracking-tight" title={d.doc_id}>
          {d.name ?? d.title ?? shortTitle(d.doc_id)}
        </h2>
        <span className="text-xs text-faint">{meta}</span>
      </div>

      <table className="w-full max-w-md border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-[11px] text-faint">
            <th className="pb-2 font-normal">Box</th>
            {d.annotated && <th className="pb-2 text-right font-normal">IoU</th>}
            <th className="pb-2 text-right font-normal">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {BOXES.map((b) => (
            <tr key={b.key} className="border-b border-line/60">
              <td className="py-2">
                <span className="flex items-center gap-2">
                  <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: b.color }} />
                  {b.label}
                </span>
              </td>
              {d.annotated && <td className={`tabular py-2 text-right ${lowClass(d.iou?.[b.key], 0.5)}`}>{num(d.iou?.[b.key])}</td>}
              <td className={`tabular py-2 text-right text-muted ${lowClass(d.confidence?.[b.key], LOW_CONF)}`}>{num(d.confidence?.[b.key])}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="flex flex-wrap items-center gap-2.5">
        <Segmented
          label="Region"
          items={[
            { id: "header", label: "Header" },
            { id: "body", label: "Body" },
          ]}
          value={view}
          onChange={setView}
        />
        <Toggle on={showDet} set={setShowDet} label="Detected" swatch={<DetectedSwatch />} />
        {d.annotated && <Toggle on={showAnn} set={setShowAnn} label="Annotated" swatch={<AnnotatedSwatch />} />}
        {view === "body" && (
          <input
            type="range"
            min={0}
            max={1000}
            value={Math.round(pos * 1000)}
            onChange={(e) => setPos(Number(e.target.value) / 1000)}
            className="min-w-40 flex-1 accent-[#8f74ea]"
            aria-label="Position in the log body"
          />
        )}
      </div>

      <div ref={ref} className="max-h-[78vh] overflow-auto rounded-lg border border-line bg-paper">
        <div className="relative w-full">
          {loaded !== url && (
            <div className="absolute inset-x-0 top-0 flex items-center p-3 text-[#555]">
              <Spinner />
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
            {showDet &&
              BOXES.map((b) => {
                const [x0, by0, x1, by1] = d.boxes[b.key];
                return (
                  <rect
                    key={`det-${b.key}`}
                    x={x0}
                    y={by0}
                    width={x1 - x0}
                    height={by1 - by0}
                    fill={b.color}
                    fillOpacity={0.1}
                    stroke={b.color}
                    strokeWidth={2.5}
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
            {showAnn &&
              d.annotated &&
              BOXES.map((b) => {
                const a = d.annotated![b.key];
                if (!a) return null;
                const r = { x: a[0], y: a[1], width: a[2] - a[0], height: a[3] - a[1], fill: "none", vectorEffect: "non-scaling-stroke" as const };
                return (
                  <g key={`ann-${b.key}`}>
                    <rect {...r} stroke="#FFFFFF" strokeWidth={ANN_HALO} strokeOpacity={0.9} />
                    <rect {...r} stroke="#111111" strokeWidth={ANN_STROKE} strokeDasharray={ANN_DASH} />
                  </g>
                );
              })}
          </svg>
        </div>
      </div>
    </div>
  );
}

function DetectedSwatch() {
  return (
    <svg width="18" height="12" viewBox="0 0 18 12" aria-hidden="true">
      <rect x="1.25" y="1.25" width="15.5" height="9.5" rx="1" fill="#B79CFF" fillOpacity="0.25" stroke="#B79CFF" strokeWidth="2" />
    </svg>
  );
}

function AnnotatedSwatch() {
  return (
    <svg width="18" height="12" viewBox="0 0 18 12" aria-hidden="true">
      <rect x="0" y="0" width="18" height="12" rx="2" fill="#FBFBF9" />
      <rect x="2.5" y="2.5" width="13" height="7" fill="none" stroke="#111111" strokeWidth="2" strokeDasharray="4 2.5" />
    </svg>
  );
}

function Toggle({ on, set, label, swatch }: { on: boolean; set: (v: boolean) => void; label: string; swatch: ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={() => set(!on)}
      className={`flex h-[38px] items-center gap-2 rounded-lg px-3 text-xs transition-colors ${on ? "bg-line-2 text-ink" : "bg-panel text-faint hover:text-ink"}`}
    >
      <span className={on ? "" : "opacity-40"}>{swatch}</span>
      {label}
    </button>
  );
}
