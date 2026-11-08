import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Check, FileJson, FileImage, X } from "lucide-react";
import { startRun, useRun } from "../api";
import type { BoxMode, RunBoxes } from "../api";
import { fmt, memory, seconds } from "../format";
import { ErrorNote, Segmented } from "./ui";

const STEP_LABEL: Record<string, string> = {
  load: "Load image",
  layout: "Find the three boxes",
  scale: "Read depth scale",
  segment: "Find bed boundaries",
  cluster: "Group similar beds",
  legend: "Match legend",
  assemble: "Assemble intervals",
  write: "Write CSV",
  tiles: "Cut scan tiles",
  ingest: "Add to dashboard",
};
const ORDER = ["load", "layout", "scale", "segment", "cluster", "legend", "assemble", "write", "tiles", "ingest"];
const isImage = (f: File) => /\.(tiff?|png|jpe?g)$/i.test(f.name);
const BOX_MODES: { id: BoxMode; label: string }[] = [
  { id: "auto", label: "Box file" },
  { id: "detect", label: "Layout detector" },
];
const BOX_NAMES: Record<string, string> = {
  interpreted_lithology_column: "lithology",
  lithology_legend: "legend",
  depth_column: "depth",
};

export default function RunDrawer({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [drag, setDrag] = useState(false);
  const [mode, setMode] = useState<BoxMode>("auto");
  const input = useRef<HTMLInputElement>(null);
  const { data: run } = useRun(job);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const busy = !!job && run?.state !== "done" && run?.state !== "error";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !busy && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  useEffect(() => {
    if (run?.state === "done") {
      qc.invalidateQueries({ queryKey: ["logs"] });
      qc.invalidateQueries({ queryKey: ["insights"] });
      qc.invalidateQueries({ queryKey: ["meta"] });
    }
  }, [run?.state, qc]);

  const add = (list: FileList | null) => {
    if (!list) return;
    const incoming = [...list].filter((f) => isImage(f) || f.name.toLowerCase().endsWith(".json"));
    setFiles((prev) => {
      const keep = prev.filter((p) => !incoming.some((n) => isImage(n) === isImage(p)));
      return [...keep, ...incoming].slice(0, 2);
    });
    setError(null);
  };

  const go = async () => {
    setError(null);
    try {
      setJob(await startRun(files, boxes ? mode : "auto"));
    } catch (e) {
      setError(e);
    }
  };

  const image = files.find(isImage);
  const boxes = files.find((f) => !isImage(f));
  const current = run?.state === "ingesting" ? "ingest" : run?.step;
  const doneSteps = run?.state === "done" ? new Set(ORDER) : new Set(Object.keys(run?.steps ?? {}));

  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/55" onClick={() => !busy && onClose()} aria-hidden="true" />
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Run a scan"
        className="absolute bottom-0 right-0 top-0 flex w-[min(480px,100%)] flex-col gap-7 border-l border-line-2 bg-panel p-8"
      >
        <div className="flex items-start gap-3">
          <div className="flex flex-col gap-1.5">
            <h1 className="text-lg font-semibold">Run a scan</h1>
            <p className="text-sm leading-relaxed text-muted">The pipeline runs on this laptop. Results appear in all three views.</p>
          </div>
          <button
            type="button"
            aria-label="Close"
            disabled={busy}
            onClick={onClose}
            className="ml-auto flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-hover disabled:opacity-40"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        {!job && (
          <>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                add(e.dataTransfer.files);
              }}
              className={`flex h-32 flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed text-center text-sm transition-colors ${
                drag ? "border-select bg-hover" : "border-line-2"
              }`}
            >
              <span className="text-ink">Drop a scan</span>
              <span className="text-muted">
                .tif or .png, optional .json, or{" "}
                <button type="button" onClick={() => input.current?.click()} className="text-ink underline underline-offset-2">
                  browse
                </button>
              </span>
              <input ref={input} type="file" multiple hidden accept=".tif,.tiff,.png,.jpg,.jpeg,.json" onChange={(e) => add(e.target.files)} />
            </div>
            <div className="flex flex-col gap-3 text-sm">
              <FileRow icon={<FileImage size={15} />} file={image} empty="No scan yet" />
              <FileRow icon={<FileJson size={15} />} file={boxes} empty="No box file: the layout detector finds the boxes" />
            </div>
            {boxes && (
              <div className="flex items-center gap-3 text-sm">
                <span className="text-muted">Boxes from</span>
                <Segmented label="Boxes from" items={BOX_MODES} value={mode} onChange={setMode} tone="bg-panel-2" />
              </div>
            )}
            {error != null && <ErrorNote error={error} />}
            <div className="mt-auto flex items-center justify-end gap-2.5">
              <button type="button" onClick={onClose} className="h-10 rounded-lg px-3.5 text-sm text-muted hover:bg-hover hover:text-ink">
                Cancel
              </button>
              <button
                type="button"
                disabled={!image}
                onClick={go}
                className="h-10 rounded-lg bg-accent px-4 text-sm font-medium text-white transition-colors hover:bg-accent-hi disabled:bg-panel-2 disabled:text-faint"
              >
                Run pipeline
              </button>
            </div>
          </>
        )}

        {job && (
          <>
            <div className="flex flex-col gap-1">
              <span className="truncate font-mono text-xs text-ink">{run?.doc_id ?? image?.name}</span>
              {run?.size && <span className="text-xs text-faint">{`${fmt(run.size[0])} × ${fmt(run.size[1])} px`}</span>}
            </div>
            <div className="h-px bg-line" />
            <div className="flex flex-col gap-3.5">
              <span className="text-xs text-faint">Progress</span>
              {ORDER.map((s) => {
                const done = doneSteps.has(s) && s !== current;
                const active = s === current && run?.state !== "done";
                return (
                  <div key={s} className="flex items-center gap-3 text-sm">
                    <span
                      aria-hidden="true"
                      className={`h-2 w-2 rounded-full border ${
                        done ? "border-[#6fbf7a] bg-[#6fbf7a]" : active ? "pulse border-select bg-select" : "border-line-2"
                      }`}
                    />
                    <span className={done ? "text-muted" : active ? "text-ink" : "text-faint"}>{STEP_LABEL[s]}</span>
                    <span className="ml-auto font-mono text-xs text-faint">
                      {run?.steps[s] != null ? `${fmt(run.steps[s], 2)} s` : active ? "running" : ""}
                    </span>
                  </div>
                );
              })}
            </div>
            {run?.state === "error" && <ErrorNote error={run.error} />}
            <div className="mt-auto flex flex-col gap-4">
              {run?.state === "done" && (
                <div className="flex flex-col gap-2">
                  <p className="flex items-center gap-2 text-sm text-muted">
                    <Check size={15} className="text-[#6fbf7a]" aria-hidden="true" />
                    Pipeline {seconds(run.pipeline_seconds)}, peak memory {memory(run.peak_rss_mb)}
                  </p>
                  {run.boxes && <BoxesNote boxes={run.boxes} />}
                </div>
              )}
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-xs text-faint">{run ? `${fmt(run.elapsed, 1)} s elapsed` : ""}</span>
                <button
                  type="button"
                  disabled={busy}
                  onClick={onClose}
                  className="ml-auto h-10 rounded-lg px-3.5 text-sm text-muted hover:bg-hover hover:text-ink disabled:opacity-40"
                >
                  Close
                </button>
                <button
                  type="button"
                  disabled={run?.state !== "done" || !run.log_id}
                  onClick={() => {
                    onClose();
                    navigate(`/csv/${run!.log_id}`);
                  }}
                  className="h-10 rounded-lg bg-accent px-4 text-sm font-medium text-white transition-colors hover:bg-accent-hi disabled:bg-panel-2 disabled:text-faint"
                >
                  Open result
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function BoxesNote({ boxes }: { boxes: RunBoxes }) {
  if (boxes.source === "json") return <p className="text-sm text-muted">Boxes from the box file.</p>;
  const iou = boxes.iou_vs_json;
  const low = (boxes.low_confidence ?? []).map((k) => BOX_NAMES[k] ?? k);
  return (
    <div className="flex flex-col gap-1 text-sm text-muted">
      <p>Boxes found by the layout detector.</p>
      {iou && (
        <p className="text-xs text-faint">
          Overlap with the box file:{" "}
          {Object.keys(BOX_NAMES)
            .filter((k) => iou[k] != null)
            .map((k) => `${BOX_NAMES[k]} ${fmt(iou[k], 2)}`)
            .join(", ")}
        </p>
      )}
      {low.length > 0 && <p className="text-xs text-faint">Low confidence: {low.join(", ")}. Check them on the Image page.</p>}
    </div>
  );
}

function FileRow({ icon, file, empty }: { icon: React.ReactNode; file?: File; empty: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className={file ? "text-[#6fbf7a]" : "text-faint"} aria-hidden="true">
        {file ? <Check size={15} /> : icon}
      </span>
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className={`truncate ${file ? "font-mono text-xs text-ink" : "text-faint"}`}>{file ? file.name : empty}</span>
        {file && <span className="text-xs text-faint">{memory(file.size / 1e6)}</span>}
      </div>
    </div>
  );
}
