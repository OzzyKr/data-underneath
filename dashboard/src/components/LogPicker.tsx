import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ChevronDown, Search } from "lucide-react";
import { useLogs, type LogSummary } from "../api";
import { logSubtitle } from "../format";
import { usePopover } from "./ui";

export default function LogPicker({ current }: { current?: string }) {
  const { data: logs } = useLogs();
  const { open, setOpen, ref } = usePopover();
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const page = pathname.startsWith("/image") ? "image" : "csv";
  const selected = logs?.find((l) => l.log_id === current);

  const groups = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const hit = (l: LogSummary) => {
      const hay = `${l.title} ${l.well_group} ${l.doc_id}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    };
    const out = new Map<string, LogSummary[]>();
    for (const l of logs ?? []) {
      if (!hit(l)) continue;
      const key = l.source === "live" ? "Run from the dashboard" : l.well_group;
      out.set(key, [...(out.get(key) ?? []), l]);
    }
    return [...out.entries()];
  }, [logs, q]);

  const pick = (id: string) => {
    setOpen(false);
    setQ("");
    navigate(`/${page}/${id}`);
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex h-10 max-w-[min(560px,70vw)] items-center gap-2.5 rounded-lg border border-line-2 px-3 text-sm text-ink transition-colors hover:bg-hover"
      >
        <span className="truncate">{selected ? selected.title : "Choose a log"}</span>
        {selected && <span className="hidden truncate text-faint sm:inline">{selected.scale ?? selected.depth_ref ?? ""}</span>}
        <ChevronDown size={14} className="shrink-0 text-faint" aria-hidden="true" />
      </button>
      {open && (
        <div className="absolute left-0 top-12 z-30 flex max-h-[70vh] w-[min(460px,90vw)] flex-col overflow-hidden rounded-xl border border-line-2 bg-panel shadow-2xl shadow-black/50">
          <label className="flex items-center gap-2 border-b border-line px-3.5 py-3">
            <Search size={14} className="text-faint" aria-hidden="true" />
            <span className="sr-only">Search logs</span>
            <input
              autoFocus
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={`Search ${logs?.length ?? ""} logs`}
              className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-faint"
            />
          </label>
          <div role="listbox" className="overflow-y-auto py-1.5">
            {groups.map(([group, items]) => (
              <div key={group} className="py-1">
                <div className="px-3.5 pb-1 pt-2 text-[11px] font-medium text-faint">{group}</div>
                {items.map((l) => (
                  <button
                    key={l.log_id}
                    role="option"
                    aria-selected={l.log_id === current}
                    type="button"
                    onClick={() => pick(l.log_id)}
                    className={`flex w-full flex-col gap-0.5 px-3.5 py-2 text-left transition-colors hover:bg-hover ${
                      l.log_id === current ? "bg-panel-2" : ""
                    }`}
                  >
                    <span className="flex items-center gap-2 text-sm text-ink">
                      {l.title}
                      {l.source === "live" && <span className="rounded bg-panel-2 px-1.5 py-px text-[10px] text-select">live</span>}
                    </span>
                    <span className="text-xs text-faint">{logSubtitle(l)}</span>
                  </button>
                ))}
              </div>
            ))}
            {!groups.length && <p className="px-3.5 py-6 text-sm text-faint">No log matches “{q}”.</p>}
          </div>
        </div>
      )}
    </div>
  );
}
