import type { LogSummary } from "./api";

export const fmt = (n: number | null | undefined, digits = 0): string =>
  n == null || Number.isNaN(n)
    ? "–"
    : n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const pct = (x: number | null | undefined, digits = 0): string => (x == null ? "–" : `${fmt(100 * x, digits)}%`);

export const memory = (mb: number | null | undefined): string =>
  mb == null ? "–" : mb >= 1000 ? `${fmt(mb / 1000, 1)} GB` : `${fmt(mb)} MB`;

export const seconds = (s: number | null | undefined): string => (s == null ? "–" : `${fmt(s, 1)} s`);

export function logSubtitle(l: Pick<LogSummary, "scale" | "depth_ref" | "kind" | "top_ft" | "bottom_ft">): string {
  const range = l.top_ft != null && l.bottom_ft != null ? `${fmt(l.top_ft)} – ${fmt(l.bottom_ft)} ft` : null;
  const kind = l.kind === "black and white" ? "black and white" : l.kind === "colour" ? "colour" : null;
  return [l.scale, l.depth_ref, kind, range].filter(Boolean).join(" · ");
}

export function lastLog(): string | null {
  try {
    return localStorage.getItem("du:lastLog");
  } catch {
    return null;
  }
}

export function rememberLog(id: string) {
  try {
    localStorage.setItem("du:lastLog", id);
  } catch {
    /* storage unavailable: fine */
  }
}
