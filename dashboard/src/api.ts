import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export type LogSummary = {
  log_id: string;
  doc_id: string;
  title: string;
  well_group: string;
  source: "batch" | "live";
  kind: string;
  depth_ref: string | null;
  scale: string | null;
  top_ft: number | null;
  bottom_ft: number | null;
  n_intervals: number;
  n_unresolved: number;
};

export type Tiles = {
  view: [number, number, number, number];
  scale: number;
  w: number;
  h: number;
  tile_h: number;
  n: number;
  legend: { box: number[]; w: number; h: number } | null;
  col?: { box: [number, number, number, number]; scale: number; w: number; h: number; tile_h: number; n: number };
};

export type Boxes = {
  depth_column: [number, number, number, number];
  interpreted_lithology_column: [number, number, number, number];
  lithology_legend: [number, number, number, number];
};

export type Rock = {
  rock: string;
  color: string;
  family: string;
  beds: number;
  feet: number | null;
  clusters: number[];
  unresolved: boolean;
  first: number;
};

export type LegendEntry = {
  idx: number;
  label: string;
  is_rock: boolean;
  used: boolean;
  beds: number;
  cluster_id: number | null;
  color: string | null;
  rgb: string | null;
  swatch_url: string | null;
};

export type LogDetail = LogSummary & {
  feet: number;
  n_rocks: number;
  mean_confidence: number | null;
  px_per_ft: number | null;
  scale_points: number;
  scale_confidence: number | null;
  seconds: number | null;
  peak_rss_mb: number | null;
  timings: Record<string, number> | null;
  warnings: string[];
  image_w: number | null;
  image_h: number | null;
  boxes: Boxes | null;
  tiles: Tiles | null;
  rocks: Rock[];
  legend: LegendEntry[];
};

export type Interval = {
  idx: number;
  top_depth: number | null;
  bottom_depth: number | null;
  thickness_ft: number | null;
  top_y: number;
  bottom_y: number;
  cluster_id: number | null;
  lithology: string;
  rock: string;
  rock_family: string;
  unresolved: boolean;
  confidence: number | null;
  notes: string;
  color: string;
};

export type Note = { text: string; source: "ai" | "computed"; provider: string | null; model: string | null; cached?: boolean; at?: string };
export type MixRow = { log_id: string; well: string; title: string; feet: number; shares: { family: string; share: number; ft: number }[] };
export type CorrLog = { log_id: string; well: string; title: string; runs: { top: number; bottom: number; family: string }[] };
export type Insights = {
  headline: {
    logs: number;
    beds: number;
    feet: number;
    wells: number;
    seconds_per_log: number | null;
    peak_mb: number | null;
    rock_names: number;
    unresolved_share: number | null;
  };
  mix: MixRow[];
  correlation: { logs: CorrLog[]; candidates: { log_id: string; well: string; title: string }[] };
  families: { family: string; color: string }[];
  notes: { mix: Note; correlation: Note };
  suggested: string[];
};

export type Answer = {
  question: string;
  answer: string;
  chart: { type: "bar" | "line" | "table"; x: string; y: string[]; title: string; columns: string[]; rows: unknown[][] } | null;
  queries: string[];
  provider: string;
  model: string | null;
  seconds: number;
  at: string;
  cached: boolean;
};

export type ProviderInfo = { name: string; label: string; available: boolean; model: string | null; reason: string };
export type AiStatus = { active: string; model: string | null; providers: ProviderInfo[] };

export type RunStatus = {
  id: string;
  doc_id: string;
  state: "queued" | "running" | "ingesting" | "done" | "error";
  step: string | null;
  steps: Record<string, number>;
  order: string[];
  elapsed: number;
  pipeline_seconds: number | null;
  peak_rss_mb: number | null;
  size: [number, number] | null;
  log_id: string | null;
  error: string | null;
};

export type Meta = {
  logs: number;
  include_holdout?: boolean;
  holdout_hidden?: number;
  pipeline?: { mean_seconds_per_doc: number; max_peak_rss_mb: number; commit: string };
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const post = <T,>(url: string, body: unknown) =>
  request<T>(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const useMeta = () => useQuery({ queryKey: ["meta"], queryFn: () => request<Meta>("/api/meta") });
export const useLogs = () => useQuery({ queryKey: ["logs"], queryFn: () => request<LogSummary[]>("/api/logs") });
export const useLog = (id?: string) =>
  useQuery({ queryKey: ["log", id], queryFn: () => request<LogDetail>(`/api/logs/${id}`), enabled: !!id });
export const useIntervals = (id?: string) =>
  useQuery({ queryKey: ["intervals", id], queryFn: () => request<Interval[]>(`/api/logs/${id}/intervals`), enabled: !!id });
export const useInsights = (corr: string[] | null) =>
  useQuery({
    queryKey: ["insights", corr?.join(",") ?? ""],
    queryFn: () => request<Insights>(`/api/insights${corr?.length ? `?correlation=${corr.join(",")}` : ""}`),
    placeholderData: (prev) => prev,
  });
export const useAi = () => useQuery({ queryKey: ["ai"], queryFn: () => request<AiStatus>("/api/ai") });

export function useSetProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (provider: string) => post<AiStatus>("/api/ai", { provider }),
    onSuccess: (data) => qc.setQueryData(["ai"], data),
  });
}

export function useRegenerateNotes(corr: string[] | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => post<Insights>("/api/insights/notes", { correlation: corr }),
    onSuccess: (data) => qc.setQueryData(["insights", corr?.join(",") ?? ""], data),
  });
}

export const useAsk = () => useMutation({ mutationFn: (question: string) => post<Answer>("/api/ask", { question }) });

export async function startRun(files: File[]): Promise<string> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  const r = await request<{ id: string }>("/api/runs", { method: "POST", body: fd });
  return r.id;
}

export const useRun = (id: string | null) =>
  useQuery({
    queryKey: ["run", id],
    queryFn: () => request<RunStatus>(`/api/runs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ["done", "error"].includes(q.state.data.state) ? false : 600),
  });
