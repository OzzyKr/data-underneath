import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronDown, Download, Search } from "lucide-react";
import { useIntervals, useLog, type Interval, type Rock } from "../api";
import { fmt, logSubtitle, memory, pct, rememberLog, seconds } from "../format";
import { Dot, ErrorNote, Stat } from "../components/ui";

const COLS = "grid grid-cols-[1fr_1fr_1.5fr_0.9fr_0.9fr_0.7fr_0.8fr] gap-x-4";

export default function CsvPage() {
  const { logId } = useParams();
  const [params] = useSearchParams();
  const selected = params.get("i") != null ? Number(params.get("i")) : null;
  const { data: log, error } = useLog(logId);
  const { data: rows } = useIntervals(logId);
  const navigate = useNavigate();
  const [filter, setFilter] = useState("");
  const [rock, setRock] = useState("");

  useEffect(() => {
    if (logId) rememberLog(logId);
    setFilter("");
    setRock("");
  }, [logId]);

  const filtered = useMemo(() => {
    const f = filter.trim().toLowerCase();
    return (rows ?? []).filter(
      (r) =>
        (!rock || r.rock === rock) &&
        (!f ||
          r.rock.toLowerCase().includes(f) ||
          r.notes.toLowerCase().includes(f) ||
          String(r.top_depth ?? "").startsWith(f) ||
          String(r.bottom_depth ?? "").startsWith(f)),
    );
  }, [rows, filter, rock]);

  const scroller = useRef<HTMLDivElement>(null);
  const virt = useVirtualizer({ count: filtered.length, getScrollElement: () => scroller.current, estimateSize: () => 44, overscan: 12 });

  useEffect(() => {
    if (selected == null || !rows) return;
    const at = filtered.findIndex((r) => r.idx === selected);
    if (at >= 0) virt.scrollToIndex(at, { align: "center" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, rows]);

  if (error) return <div className="p-12"><ErrorNote error={error} /></div>;
  if (!log) return null;
  const totalFeet = log.rocks.reduce((s, r) => s + (r.feet ?? 0), 0) || 1;

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-8 px-6 pb-14 pt-10 lg:px-12">
      <div className="flex flex-col gap-1.5">
        <h1 className="text-2xl font-semibold tracking-tight">{log.title}</h1>
        <span className="text-sm text-muted">{logSubtitle(log)}</span>
        <span className="truncate font-mono text-xs text-faint" title={log.doc_id}>
          {log.doc_id}
        </span>
      </div>

      <div className="flex flex-wrap gap-x-12 gap-y-5">
        <Stat label="Intervals" value={fmt(log.n_intervals)} />
        <Stat label="Rock types" value={fmt(log.n_rocks)} />
        <Stat label="Feet interpreted" value={fmt(log.feet)} />
        <Stat
          label="Unresolved"
          value={`${fmt(log.n_unresolved)} beds`}
          hint="Beds computer vision could not name with confidence; the CSV says 'Unresolved' instead of guessing."
        />
        <Stat
          label="Processing"
          value={`${seconds(log.seconds)} · ${memory(log.peak_rss_mb)}`}
          hint="Pipeline time and peak memory for this scan, on a laptop CPU."
        />
      </div>

      <Composition rocks={log.rocks} total={totalFeet} />

      <section aria-label="Intervals" className="overflow-hidden rounded-xl border border-line">
        <div className="flex flex-wrap items-center gap-2.5 border-b border-line px-4 py-3">
          <label className="flex h-9 w-60 max-w-full items-center gap-2 rounded-lg bg-panel px-2.5">
            <Search size={14} className="text-faint" aria-hidden="true" />
            <span className="sr-only">Filter rows</span>
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by rock or depth"
              className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-faint"
            />
          </label>
          <label className="relative flex h-9 items-center rounded-lg bg-panel">
            <span className="sr-only">Rock</span>
            <select
              value={rock}
              onChange={(e) => setRock(e.target.value)}
              className="h-9 appearance-none bg-transparent pl-3 pr-8 text-sm text-ink outline-none"
            >
              <option value="">All rocks</option>
              {log.rocks.map((r) => (
                <option key={r.rock} value={r.rock}>
                  {r.rock}
                </option>
              ))}
            </select>
            <ChevronDown size={13} className="pointer-events-none absolute right-2.5 text-faint" aria-hidden="true" />
          </label>
          <span className="text-xs text-faint">{filtered.length === rows?.length ? "" : `${fmt(filtered.length)} shown`}</span>
          <a
            href={`/api/logs/${log.log_id}/csv`}
            download
            className="ml-auto flex h-9 items-center gap-2 rounded-lg px-3 text-sm text-muted no-underline transition-colors hover:bg-hover hover:text-ink"
          >
            <Download size={14} aria-hidden="true" />
            Download CSV
          </a>
        </div>

        <div ref={scroller} className="h-[min(62vh,640px)] min-h-[360px] overflow-auto">
          <div className="min-w-[820px]">
            <div className={`${COLS} sticky top-0 z-10 bg-bg px-5 py-2.5 font-mono text-[11px] text-faint`}>
              <span>top_depth</span>
              <span>bottom_depth</span>
              <span>lithology</span>
              <span>top_y</span>
              <span>bottom_y</span>
              <span>cluster_id</span>
              <span className="text-right">confidence</span>
            </div>
            <div className="relative" style={{ height: virt.getTotalSize() }}>
              {virt.getVirtualItems().map((v) => (
                <Row
                  key={v.key}
                  r={filtered[v.index]}
                  top={v.start}
                  active={filtered[v.index].idx === selected}
                  onOpen={() => navigate(`/image/${log.log_id}?i=${filtered[v.index].idx}`)}
                />
              ))}
            </div>
          </div>
        </div>
        <div className="border-t border-line px-5 py-3 text-xs text-faint">
          {fmt(rows?.length ?? 0)} rows · click a row to see the bed on the scan
        </div>
      </section>
    </div>
  );
}

function Row({ r, top, active, onOpen }: { r: Interval; top: number; active: boolean; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      style={{ transform: `translateY(${top}px)` }}
      className={`${COLS} tabular absolute left-0 right-0 top-0 h-11 items-center border-t border-line/70 px-5 text-left font-mono text-[13px] text-ink transition-colors hover:bg-hover ${
        active ? "bg-panel-2" : ""
      }`}
    >
      <span>{raw(r.top_depth)}</span>
      <span>{raw(r.bottom_depth)}</span>
      <span className="flex min-w-0 items-center gap-2.5 font-sans text-sm" title={r.notes || undefined}>
        <Dot color={r.color} unresolved={r.unresolved} />
        <span className={`truncate ${r.unresolved ? "text-rose" : ""}`}>{r.lithology}</span>
        {r.unresolved && r.notes && <span className="truncate text-xs text-faint">{shortNote(r.notes)}</span>}
      </span>
      <span className="text-muted">{r.top_y}</span>
      <span className="text-muted">{r.bottom_y}</span>
      <span className="text-muted">{r.cluster_id ?? ""}</span>
      <span className="text-right text-muted">{fmt(r.confidence, 2)}</span>
    </button>
  );
}

/** A depth as the CSV writes it (no thousands separators). */
const raw = (v: number | null) => (v == null ? "" : String(v));

function shortNote(note: string): string {
  const m = note.match(/differentiate:\s*(.*)$/i);
  return m ? m[1] : note;
}

function Composition({ rocks, total }: { rocks: Rock[]; total: number }) {
  const shown = rocks.filter((r) => (r.feet ?? 0) > 0);
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex h-1.5 gap-0.5 overflow-hidden rounded-full" role="img" aria-label="Share of logged feet by rock">
        {shown.map((r) => (
          <div
            key={r.rock}
            title={`${r.rock}: ${pct((r.feet ?? 0) / total, 1)}`}
            className={r.unresolved ? "hatch" : ""}
            style={{ flex: `${r.feet} 1 0`, backgroundColor: r.color }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {shown.slice(0, 10).map((r) => (
          <span key={r.rock} className="flex items-center gap-2 text-[13px] text-muted">
            <Dot color={r.color} unresolved={r.unresolved} />
            <span className="text-ink">{r.rock}</span>
            {pct((r.feet ?? 0) / total)}
          </span>
        ))}
        {shown.length > 10 && <span className="text-[13px] text-faint">+{shown.length - 10} more</span>}
      </div>
    </div>
  );
}
