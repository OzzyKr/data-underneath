import { Check, ChevronDown } from "lucide-react";
import { useAi, useSetProvider } from "../api";
import { usePopover } from "./ui";

export default function ProviderSwitch() {
  const { data } = useAi();
  const set = useSetProvider();
  const { open, setOpen, ref } = usePopover();
  const active = data?.providers.find((p) => p.name === data.active);
  const label = active ? (active.name === "none" ? active.label : `${active.label} · ${data?.model ?? ""}`) : "AI";

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex h-10 items-center gap-2 rounded-lg border border-line-2 px-3 text-sm text-muted transition-colors hover:bg-hover hover:text-ink"
      >
        <span className={`h-1.5 w-1.5 rounded-full ${data?.active === "none" ? "bg-faint" : "bg-select"}`} aria-hidden="true" />
        AI: {label}
        <ChevronDown size={13} className="text-faint" aria-hidden="true" />
      </button>
      {open && data && (
        <div role="menu" className="absolute right-0 top-12 z-30 w-80 rounded-xl border border-line-2 bg-panel p-1.5 shadow-2xl shadow-black/50">
          {data.providers.map((p) => (
            <button
              key={p.name}
              role="menuitemradio"
              aria-checked={p.name === data.active}
              type="button"
              disabled={!p.available || set.isPending}
              onClick={() => {
                set.mutate(p.name);
                setOpen(false);
              }}
              className="flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 text-left transition-colors enabled:hover:bg-hover disabled:cursor-not-allowed"
            >
              <span className="mt-0.5 w-4 text-select">{p.name === data.active && <Check size={14} aria-hidden="true" />}</span>
              <span className="flex flex-col gap-0.5">
                <span className={`text-sm ${p.available ? "text-ink" : "text-faint"}`}>{p.label}</span>
                <span className="text-xs text-faint">
                  {p.available ? (p.model ?? "Answers and notes from earlier runs") : p.reason}
                </span>
              </span>
            </button>
          ))}
          <p className="border-t border-line px-3 pb-1.5 pt-2.5 text-xs leading-relaxed text-faint">
            Keys live in <code className="font-mono">.env</code>: ANTHROPIC_API_KEY, or OPENAI_API_KEY + OPENAI_MODEL
            (+ OPENAI_BASE_URL for any compatible endpoint).
          </p>
        </div>
      )}
    </div>
  );
}
