import { useMemo, useState } from "react";
import { ChevronRight, Sparkles } from "lucide-react";
import { useAi, useAsk, useInsights, useRegenerateNotes, type Insights, type Note } from "../api";
import { fmt, memory, pct } from "../format";
import { Dot, ErrorNote, Spinner, Stat, usePopover } from "../components/ui";
import AnswerChart from "../components/AnswerChart";
import { depthTicks, shortDepth } from "../viewer/depth";

export default function InsightsPage() {
  const [corr, setCorr] = useState<string[] | null>(null);
  const { data, error } = useInsights(corr);
  const { data: ai } = useAi();
  const notes = useRegenerateNotes(corr);
  const online = !!ai && ai.active !== "none";

  if (error) return <div className="p-12"><ErrorNote error={error} /></div>;
  if (!data) return null;
  const h = data.headline;
  const colour = Object.fromEntries(data.families.map((f) => [f.family, f.color]));

  return (
    <div className="mx-auto flex max-w-[1200px] flex-col gap-12 px-6 pb-16 pt-10 lg:px-12">
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1.5">
          <h1 className="text-2xl font-semibold tracking-tight">Insights</h1>
          <span className="text-sm text-muted">{fmt(h.logs)} scanned well logs, turned into one dataset you can question.</span>
        </div>
        <button
          type="button"
          disabled={!online || notes.isPending}
          onClick={() => notes.mutate()}
          title={online ? "Ask the AI to write the notes under the charts" : "Pick an AI provider in the top bar first"}
          className="ml-auto flex h-9 items-center gap-2 rounded-lg px-3 text-[13px] text-muted transition-colors enabled:hover:bg-hover enabled:hover:text-ink disabled:opacity-50"
        >
          {notes.isPending ? <Spinner /> : <Sparkles size={14} aria-hidden="true" />}
          {notes.isPending ? "Writing notes…" : "Write chart notes with AI"}
        </button>
      </div>
      {notes.error && <ErrorNote error={notes.error} />}

      <div className="flex flex-wrap gap-x-12 gap-y-5">
        <Stat label="Logs" value={fmt(h.logs)} />
        <Stat label="Beds" value={fmt(h.beds)} />
        <Stat label="Feet of borehole" value={fmt(h.feet)} />
        <Stat label="Rock names" value={fmt(h.rock_names)} />
        <Stat label="Unresolved" value={pct(h.unresolved_share)} hint="Share of logged feet the pipeline would not name with confidence" />
        <Stat
          label="Processing"
          value={`${fmt(h.seconds_per_log, 1)} s / log`}
          hint={`Laptop CPU. Peak memory ${memory(h.peak_mb)} on the largest scan.`}
        />
      </div>

      <Ask suggested={data.suggested} online={online} />

      <div className="flex flex-wrap items-start gap-12">
        <RockMix data={data} colour={colour} note={data.notes.mix} />
        <Correlation data={data} colour={colour} note={data.notes.correlation} chosen={corr} onChoose={setCorr} />
      </div>
    </div>
  );
}

function NoteLine({ note }: { note: Note }) {
  if (!note.text) return null;
  const who = note.source === "ai" ? `AI note · ${note.model ?? note.provider}` : "Summary";
  return (
    <p className="m-0 text-[13px] leading-relaxed text-muted">
      <span className="text-faint">{who} · </span>
      {note.text}
    </p>
  );
}

function Legend({ items, colour }: { items: string[]; colour: Record<string, string> }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2">
      {items.map((f) => (
        <span key={f} className="flex items-center gap-1.5 text-xs text-muted">
          <Dot color={colour[f]} unresolved={f === "Unresolved"} />
          {f === "Salt" ? "Salt / evaporite" : f}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- ask

function Ask({ suggested, online }: { suggested: string[]; online: boolean }) {
  const [q, setQ] = useState(suggested[0] ?? "");
  const ask = useAsk();
  const [showSql, setShowSql] = useState(false);
  const submit = (question: string) => {
    setQ(question);
    setShowSql(false);
    if (question.trim()) ask.mutate(question.trim());
  };
  const a = ask.data;

  return (
    <section aria-label="Ask the data" className="flex flex-col gap-4">
      <h2 className="text-[15px] font-semibold">Ask the data</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(q);
        }}
        className="flex flex-wrap gap-2.5"
      >
        <label className="flex h-12 min-w-0 flex-[1_1_420px] items-center rounded-xl border border-line-2 bg-panel px-4">
          <span className="sr-only">Question</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Ask anything about the logs"
            className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-faint"
          />
        </label>
        <button
          type="submit"
          disabled={ask.isPending || !q.trim()}
          className="h-12 rounded-xl bg-accent px-6 text-sm font-medium text-white transition-colors hover:bg-accent-hi disabled:bg-panel-2 disabled:text-faint"
        >
          {ask.isPending ? "Thinking…" : "Ask"}
        </button>
      </form>
      <div className="flex flex-wrap gap-2">
        {suggested.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => submit(s)}
            className="h-8 rounded-full border border-line-2 px-3 text-xs text-muted transition-colors hover:bg-hover hover:text-ink"
          >
            {s}
          </button>
        ))}
      </div>
      {!online && !a && !ask.error && (
        <p className="text-xs text-faint">AI is offline: questions answered earlier still work. Pick a provider in the top bar to ask new ones.</p>
      )}
      {ask.isPending && (
        <div className="flex items-center gap-2.5 rounded-xl border border-line bg-panel p-6 text-sm text-muted">
          <Spinner /> Querying the logs…
        </div>
      )}
      {ask.error && !ask.isPending && (
        <div className="rounded-xl border border-line bg-panel p-6">
          <ErrorNote error={ask.error} />
        </div>
      )}
      {a && !ask.isPending && (
        <div className="flex flex-col gap-5 rounded-xl border border-line bg-panel p-6">
          <p className="m-0 max-w-[760px] text-sm leading-relaxed text-ink">{a.answer}</p>
          {a.chart && <AnswerChart chart={a.chart} />}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            {a.queries.length > 0 && (
              <button
                type="button"
                aria-expanded={showSql}
                onClick={() => setShowSql(!showSql)}
                className="flex h-8 items-center gap-1.5 text-xs text-muted hover:text-ink"
              >
                <ChevronRight size={12} className={`transition-transform ${showSql ? "rotate-90" : ""}`} aria-hidden="true" />
                {showSql ? "Hide" : "Show"} the SQL that answered this ({a.queries.length})
              </button>
            )}
            <span className="ml-auto text-xs text-faint">
              {a.cached ? `Cached answer from ${a.at.replace("T", " ")} · ` : ""}
              {a.model ?? a.provider} · {fmt(a.seconds, 1)} s
            </span>
          </div>
          {showSql && (
            <div className="flex flex-col gap-2">
              {a.queries.map((sql, i) => (
                <pre key={i} className="m-0 overflow-x-auto rounded-lg bg-bg p-3.5 font-mono text-xs leading-relaxed text-muted">
                  {sql}
                </pre>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------------------------- rock mix

function RockMix({ data, colour, note }: { data: Insights; colour: Record<string, string>; note: Note }) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [table, setTable] = useState(false);
  const present = data.families.map((f) => f.family).filter((f) => data.mix.some((w) => w.shares.some((s) => s.family === f)));

  return (
    <section aria-label="Rock mix by well" className="relative flex min-w-0 flex-[1_1_440px] flex-col gap-5">
      <div className="flex items-end">
        <div className="flex flex-col gap-1">
          <h2 className="text-[15px] font-semibold">Rock mix by well</h2>
          <span className="text-xs text-faint">Share of logged feet by rock family · one log per well</span>
        </div>
        <button type="button" onClick={() => setTable(!table)} className="ml-auto h-8 text-xs text-muted hover:text-ink">
          {table ? "Show chart" : "Show as table"}
        </button>
      </div>
      {table ? (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="text-left text-faint">
                <th className="py-1.5 pr-3 font-normal">Well</th>
                {present.map((f) => (
                  <th key={f} className="py-1.5 pr-3 text-right font-normal">
                    {f}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {data.mix.map((w) => (
                <tr key={w.log_id} className="border-t border-line">
                  <td className="py-1.5 pr-3 text-ink">{w.well}</td>
                  {present.map((f) => {
                    const s = w.shares.find((x) => x.family === f);
                    return (
                      <td key={f} className="py-1.5 pr-3 text-right text-muted">
                        {s ? pct(s.share) : ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="flex flex-col gap-3" onMouseLeave={() => setTip(null)}>
          {data.mix.map((w) => (
            <div key={w.log_id} className="grid grid-cols-[120px_1fr] items-center gap-3.5 text-[13px]">
              <span className="truncate text-muted" title={w.title}>
                {w.well}
              </span>
              <div className="flex h-3.5 gap-0.5 overflow-hidden rounded-[3px]">
                {w.shares.map((s) => (
                  <div
                    key={s.family}
                    className={s.family === "Unresolved" ? "hatch" : ""}
                    style={{ flex: `${s.share} 1 0`, backgroundColor: colour[s.family] }}
                    onMouseMove={(e) => {
                      const box = e.currentTarget.closest("section")!.getBoundingClientRect();
                      setTip({
                        x: e.clientX - box.left,
                        y: e.clientY - box.top,
                        text: `${w.well} · ${s.family}: ${pct(s.share, 1)} (${fmt(s.ft)} ft)`,
                      });
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <Legend items={present} colour={colour} />
      <NoteLine note={note} />
      {tip && (
        <div
          className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-line-2 bg-panel px-2.5 py-1.5 text-xs text-ink shadow-lg"
          style={{ left: tip.x + 12, top: tip.y + 12 }}
        >
          {tip.text}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------------------------- correlation

function Correlation({
  data,
  colour,
  note,
  chosen,
  onChoose,
}: {
  data: Insights;
  colour: Record<string, string>;
  note: Note;
  chosen: string[] | null;
  onChoose: (ids: string[]) => void;
}) {
  const logs = data.correlation.logs;
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const pick = usePopover();
  const H = 460;
  const [lo, hi] = useMemo(() => {
    const tops = logs.flatMap((l) => l.runs.map((r) => r.top));
    const bots = logs.flatMap((l) => l.runs.map((r) => r.bottom));
    if (!tops.length) return [0, 1];
    const a = Math.min(...tops);
    const b = Math.max(...bots);
    const padv = (b - a) * 0.02;
    return [Math.max(0, a - padv), b + padv];
  }, [logs]);
  const y = (d: number) => ((d - lo) / (hi - lo)) * H;
  const ticks = depthTicks(lo, hi, 7);
  const current = chosen ?? logs.map((l) => l.log_id);
  const present = data.families
    .map((f) => f.family)
    .filter((f) => logs.some((l) => l.runs.some((r) => r.family === f)));

  return (
    <section aria-label="Well correlation" className="relative flex min-w-0 flex-[1_1_440px] flex-col gap-5">
      <div className="flex items-end">
        <div className="flex flex-col gap-1">
          <h2 className="text-[15px] font-semibold">Well correlation</h2>
          <span className="text-xs text-faint">{logs.length} wells on one depth axis (ft)</span>
        </div>
        <div ref={pick.ref} className="relative ml-auto">
          <button type="button" aria-expanded={pick.open} onClick={() => pick.setOpen(!pick.open)} className="h-8 text-xs text-muted hover:text-ink">
            Choose wells
          </button>
          {pick.open && (
            <div className="absolute right-0 top-9 z-20 max-h-80 w-64 overflow-y-auto rounded-xl border border-line-2 bg-panel p-1.5 shadow-2xl shadow-black/50">
              {data.correlation.candidates.map((c) => {
                const on = current.includes(c.log_id);
                return (
                  <label key={c.log_id} className="flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] hover:bg-hover">
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={!on && current.length >= 6}
                      onChange={() => {
                        const next = on ? current.filter((x) => x !== c.log_id) : [...current, c.log_id];
                        if (next.length) onChoose(next);
                      }}
                      className="accent-[#8F74EA]"
                    />
                    <span className="truncate text-ink">{c.well}</span>
                    <span className="ml-auto truncate text-xs text-faint">{c.title === c.well ? "" : c.title}</span>
                  </label>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <div className="flex gap-5 overflow-x-auto" onMouseLeave={() => setTip(null)}>
        <div className="relative w-9 shrink-0" style={{ height: H, marginTop: 26 }}>
          {ticks.map((d) => (
            <span key={d} className="absolute right-0 -translate-y-1/2 font-mono text-[10px] text-faint" style={{ top: y(d) }}>
              {shortDepth(d)}
            </span>
          ))}
        </div>
        {logs.map((l) => (
          <div key={l.log_id} className="flex w-20 shrink-0 flex-col items-center gap-2.5">
            <span className="h-4 max-w-20 truncate text-[11px] text-muted" title={l.title}>
              {l.well}
            </span>
            <div className="relative w-7 rounded-[3px] bg-panel-2" style={{ height: H }}>
              {l.runs.map((r, i) => (
                <div
                  key={i}
                  className={`absolute inset-x-0 ${r.family === "Unresolved" ? "hatch" : ""}`}
                  style={{ top: y(r.top), height: Math.max(1, y(r.bottom) - y(r.top)), backgroundColor: colour[r.family] }}
                  onMouseMove={(e) => {
                    const box = e.currentTarget.closest("section")!.getBoundingClientRect();
                    setTip({
                      x: e.clientX - box.left,
                      y: e.clientY - box.top,
                      text: `${l.well} · ${r.family}: ${fmt(r.top)} – ${fmt(r.bottom)} ft`,
                    });
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <Legend items={present} colour={colour} />
      <NoteLine note={note} />
      {tip && (
        <div
          className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-line-2 bg-panel px-2.5 py-1.5 text-xs text-ink shadow-lg"
          style={{ left: tip.x + 12, top: tip.y + 12 }}
        >
          {tip.text}
        </div>
      )}
    </section>
  );
}
