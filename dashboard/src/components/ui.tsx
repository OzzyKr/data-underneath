import { useEffect, useRef, useState, type ReactNode } from "react";

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="flex flex-col gap-1" title={hint}>
      <span className="text-xs text-faint">{label}</span>
      <span className="tabular text-xl font-medium tracking-tight">{value}</span>
    </div>
  );
}

export function Dot({ color, unresolved, size = 8 }: { color: string; unresolved?: boolean; size?: number }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block shrink-0 rounded-[2px] ${unresolved ? "hatch" : ""}`}
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
