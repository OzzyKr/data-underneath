import { useEffect, useRef, useState, type ReactNode } from "react";

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="flex flex-col gap-1" title={hint}>
      <span className="text-xs text-faint">{label}</span>
      <span className="tabular text-xl font-medium tracking-tight">{value}</span>
    </div>
  );
}

/** Texture class of a bed: unresolved (nothing matches clearly) or similar types (2-3 look-alike entries). */
export function texture(r: { unresolved?: boolean; lookalike?: boolean }): string {
  return r.unresolved ? "hatch" : r.lookalike ? "hatch-light" : "";
}

/** Name shown for a bed: the look-alike entries ("Clay-shale / Sandy clay-shale") when vision could not pick one. */
export function bedName(r: { rock: string; lookalike?: boolean; alternatives?: string }): string {
  return r.lookalike && r.alternatives ? r.alternatives : r.rock;
}

export function Dot({
  color,
  unresolved,
  similar,
  size = 8,
}: {
  color: string;
  unresolved?: boolean;
  similar?: boolean;
  size?: number;
}) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block shrink-0 rounded-[2px] ${texture({ unresolved, lookalike: similar })}`}
      style={{ width: size, height: size, backgroundColor: color }}
    />
  );
}

/** Open/close state for a small popover that closes on outside click and Escape. */
export function usePopover() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return { open, setOpen, ref };
}

export function Spinner() {
  return <span className="pulse inline-block h-2 w-2 rounded-full bg-select" aria-hidden="true" />;
}

export function ErrorNote({ error }: { error: unknown }) {
  return <p className="text-sm text-rose">{error instanceof Error ? error.message : String(error)}</p>;
}

export function Segmented<T extends string | number>({
  label,
  items,
  value,
  onChange,
  tone = "bg-panel",
}: {
  label: string;
  items: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  tone?: string;
}) {
  return (
    <div role="group" aria-label={label} className={`flex rounded-lg p-[3px] ${tone}`}>
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
