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
  page?: { scale: number; w: number; h: number; tile_h: number; n: number; image: [number, number] };
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
  lookalike: boolean;
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
  boxes_source?: "json" | "detector" | null;
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
  lookalike: boolean;
  alternatives: string;
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
    similar_share: number | null;
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

/** Where a run's three boxes came from: the uploaded box file, or the layout detector (with its confidence and, when a
 * box file was uploaded too, how well the detected boxes overlap it). */
export type RunBoxes = {
  source: "json" | "detector";
  confidence?: Record<string, number>;
  low_confidence?: string[];
  iou_vs_json?: Record<string, number> | null;
};
export type BoxMode = "auto" | "detect";

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
  boxes: RunBoxes | null;
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
  try {
    return (await r.json()) as T;
  } catch {
    // e.g. a dashboard server started before the page was rebuilt answers a new endpoint with the page itself
    throw new Error(`${url} did not return data. Restart the dashboard (Ctrl+C, then make dash) and reload.`);
  }
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

export async function startRun(files: File[], boxes: BoxMode = "auto"): Promise<string> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("boxes", boxes);
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

// ------------------------------------------------------------------ layout (step 1: the box detector)

export type BoxKey = "depth_column" | "interpreted_lithology_column" | "lithology_legend";
export type BoxMap<T> = Record<BoxKey, T>;
export type BoxMetric = { page_mean: number; vendor_macro: number; "iou_gt_0.5": number; "iou_lt_0.3": number };
export type LayoutRow = {
  doc_id: string;
  vendor: string;
  split: string;
  seconds: number | null;
  iou: BoxMap<number | null> | null;
  confidence: BoxMap<number | null> | null;
};
export type LayoutUpload = {
  id: string;
  name: string;
  state: "queued" | "running" | "done" | "error";
  error: string | null;
  elapsed: number;
  seconds: number | null;
  confidence: BoxMap<number> | null;
};
export type LayoutSummary = {
  run: string | null;
  mean_seconds?: number;
  max_seconds?: number;
  metrics: { dev?: { pages: number } & BoxMap<BoxMetric>; holdout?: { pages: number } & BoxMap<BoxMetric> } | null;
  pages: LayoutRow[];
  uploads?: LayoutUpload[];
};
export type LayoutDetail = LayoutRow & {
  source: "batch" | "upload";
  name?: string;
  size: [number, number] | null;
  dpi: number | null;
  body: [number, number] | null;
  boxes: BoxMap<[number, number, number, number]>;
  annotated: BoxMap<[number, number, number, number]> | null;
  notes: string[];
};

export const useLayout = () => useQuery({ queryKey: ["layout"], queryFn: () => request<LayoutSummary>("/api/layout") });
export const useLayoutPage = (id?: string) =>
  useQuery({
    queryKey: ["layout-page", id],
    queryFn: () => request<LayoutDetail>(`/api/layout/pages/${encodeURIComponent(id!)}`),
    enabled: !!id,
    retry: false,
  });
export const layoutCropUrl = (id: string, y0: number, y1: number, w: number) =>
  `/api/layout/pages/${encodeURIComponent(id)}/crop?y0=${Math.round(y0)}&y1=${Math.round(y1)}&w=${Math.round(w)}`;

/** Upload an image for layout detection (XHR, so large TIFFs report upload progress). Resolves to the job id. */
export function startDetect(file: File, onProgress?: (pct: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/layout/detect");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(Math.round((100 * e.loaded) / e.total));
    xhr.onload = () => {
      let body: { id?: string; detail?: unknown } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.id) resolve(body.id);
      else reject(new Error(typeof body.detail === "string" ? body.detail : `upload failed (${xhr.status} ${xhr.statusText})`));
    };
    xhr.onerror = () => reject(new Error("upload failed: the dashboard server is not reachable"));
    xhr.send(fd);
  });
}

export const useDetect = (id: string | null) =>
  useQuery({
    queryKey: ["detect", id],
    queryFn: () => request<LayoutUpload>(`/api/layout/detect/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ["done", "error"].includes(q.state.data.state) ? false : 700),
  });
