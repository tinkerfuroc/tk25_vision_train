export type Ontology = {
  path: string;
  hash: string;
  prompts: string[];
  labels: string[];
  mapping: Record<string, string>;
};

export type ClipSummary = {
  clip_id: string;
  source: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  bag_path: string | null;
};

export type Health = {
  ok: boolean;
  version: string;
  sam3_loaded: boolean;
  gpu: string | null;
  dtype: string | null;
  config_path: string | null;
};

export type RealSenseStatus = {
  available: boolean;
  busy: boolean;
  recording_clip_id: string | null;
  running: boolean;
  subscribers: number;
  width: number;
  height: number;
  fps: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) detail = `${r.status}: ${body.detail}`;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export type LiveInferStatus = {
  job_id: string;
  status: JobStatus;
  error: string | null;
  frame_count: number;
};

export type LiveFrameEvent = {
  event: string;
  frame_idx: number;
  timestamp: number;
  detections: InferDetection[];
  width: number;
  height: number;
  frame_b64?: string;  // base64-encoded JPEG for display
};

export const api = {
  health: () => request<Health>("/api/healthz"),
  ontology: () => request<Ontology>("/api/ontology"),
  clips: () => request<ClipSummary[]>("/api/clips"),
  runs: () => request<RunSummary[]>("/api/runs"),
  getRun: (run_id: string) => request<RunSummary>(`/api/runs/${encodeURIComponent(run_id)}`),
  models: () => request<ModelInfo[]>("/api/runs/models"),
  startLiveInfer: (body: { weights_path: string; conf?: number; iou?: number }) =>
    request<LiveInferStatus>("/api/live-infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  getLiveInfer: (job_id: string) =>
    request<LiveInferStatus>(`/api/live-infer/${encodeURIComponent(job_id)}`),
  cancelLiveInfer: (job_id: string) =>
    request<LiveInferStatus>(`/api/live-infer/${encodeURIComponent(job_id)}`, { method: "DELETE" }),
  deleteClip: (id: string) => request<{ deleted: string }>(`/api/clips/${id}`, { method: "DELETE" }),
  importFolder: (folder_path: string) =>
    request<ClipSummary>("/api/clips/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path })
    }),
  importBag: (bag_path: string) =>
    request<ClipSummary>("/api/clips/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bag_path })
    }),
  record: (max_seconds: number) =>
    request<ClipSummary>("/api/clips/record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_seconds })
    }),
  stopRecord: () => request<{ stopped: boolean }>("/api/clips/record/stop", { method: "POST" }),
  realsenseStatus: () => request<RealSenseStatus>("/api/realsense/status"),
  clipDetail: (id: string) => request<ClipDetail>(`/api/clips/${id}/detail`),
  seedClip: (id: string) =>
    request<ClipDetail>(`/api/clips/${id}/seed`, { method: "POST" }),
  refine: (
    id: string,
    frame_idx: number,
    body: {
      track_id?: number | null;
      class_id?: number | null;
      label?: string | null;
      points?: [number, number, number][];
      box?: [number, number, number, number];
    }
  ) =>
    request<{ track_id: number; frame_idx: number; width: number; height: number }>(
      `/api/clips/${id}/frames/${frame_idx}/refine`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }
    ),
  patchTrack: (id: string, tid: number, body: { class_id?: number; label?: string }) =>
    request<ClipDetail>(`/api/clips/${id}/tracks/${tid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  deleteTrack: (id: string, tid: number) =>
    request<ClipDetail>(`/api/clips/${id}/tracks/${tid}`, { method: "DELETE" }),
  trackAt: (id: string, frameIdx: number, x: number, y: number) => {
    const qs = new URLSearchParams({ x: String(x), y: String(y) }).toString();
    return request<TrackAtResult>(
      `/api/clips/${encodeURIComponent(id)}/frames/${frameIdx}/track-at?${qs}`
    );
  },
  getSam3Config: () => request<Sam3Config>("/api/sam3/config"),
  setSam3Config: (score_mode: Sam3ScoreMode) =>
    request<Sam3Config>("/api/sam3/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ score_mode })
    }),
  cleanupClip: (id: string) =>
    request<CleanupResponse>(`/api/clips/${encodeURIComponent(id)}/cleanup`, {
      method: "POST"
    }),
  deleteFrame: (id: string, idx: number) =>
    request<ClipDetail>(`/api/clips/${encodeURIComponent(id)}/frames/${idx}`, {
      method: "DELETE"
    }),
  restoreFrame: (id: string, idx: number) =>
    request<ClipDetail>(`/api/clips/${encodeURIComponent(id)}/frames/${idx}/restore`, {
      method: "POST"
    }),
  pruneFrames: (id: string, from_idx: number) =>
    request<ClipDetail>(`/api/clips/${encodeURIComponent(id)}/frames/prune`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_idx })
    }),
  propagate: (
    id: string,
    body: { start?: number; end?: number | null; chunk_size?: number; chunk_overlap?: number; respect_edits?: boolean }
  ) =>
    request<PropagateResponse>(`/api/clips/${encodeURIComponent(id)}/propagate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  cancelPropagate: (id: string, jobId: string) =>
    request<PropagateStatus>(`/api/clips/${encodeURIComponent(id)}/propagate/${jobId}`, {
      method: "DELETE"
    }),
  exportClip: (
    id: string,
    body: { run_id: string; train_ratio?: number; per_clip_split?: boolean; seed?: number; overwrite?: boolean }
  ) =>
    request<ExportResponse>(`/api/clips/${encodeURIComponent(id)}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  exportRun: (body: {
    run_id: string;
    clip_ids?: string[] | null;
    train_ratio?: number;
    per_clip_split?: boolean;
    seed?: number;
    overwrite?: boolean;
  }) =>
    request<ExportResponse>(`/api/runs/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  augmentRun: (run_id: string, body: { multiplier?: number | null; seed?: number }) =>
    request<AugmentResponse>(`/api/runs/${encodeURIComponent(run_id)}/augment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id, ...body })
    }),
  startTrain: (
    run_id: string,
    body: { base_weights?: string; epochs?: number; imgsz?: number; batch?: number; patience?: number; device?: string }
  ) =>
    request<TrainStatus>(`/api/runs/${encodeURIComponent(run_id)}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  listTrainJobs: (run_id: string) =>
    request<TrainStatus[]>(`/api/runs/${encodeURIComponent(run_id)}/train`),
  getTrain: (run_id: string, job_id: string) =>
    request<TrainStatus>(`/api/runs/${encodeURIComponent(run_id)}/train/${encodeURIComponent(job_id)}`),
  cancelTrain: (run_id: string, job_id: string) =>
    request<TrainStatus>(`/api/runs/${encodeURIComponent(run_id)}/train/${encodeURIComponent(job_id)}`, {
      method: "DELETE"
    }),
  startInfer: (
    run_id: string,
    clip_id: string,
    body: { weights_path: string; conf?: number; iou?: number }
  ) =>
    request<InferStatus>(
      `/api/runs/${encodeURIComponent(run_id)}/infer/${encodeURIComponent(clip_id)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }
    ),
  startInferNoRun: (
    clip_id: string,
    body: { weights_path: string; conf?: number; iou?: number }
  ) =>
    request<InferStatus>(
      `/api/infer/${encodeURIComponent(clip_id)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }
    ),
  getInfer: (run_id: string, clip_id: string, job_id: string) =>
    request<InferStatus>(
      `/api/runs/${encodeURIComponent(run_id)}/infer/${encodeURIComponent(clip_id)}/${encodeURIComponent(job_id)}`
    ),
  cancelInfer: (run_id: string, clip_id: string, job_id: string) =>
    request<InferStatus>(
      `/api/runs/${encodeURIComponent(run_id)}/infer/${encodeURIComponent(clip_id)}/${encodeURIComponent(job_id)}`,
      { method: "DELETE" }
    ),
  getPredictions: (
    run_id: string,
    clip_id: string,
    opts?: { frame?: number; from_idx?: number; to_idx?: number }
  ) => {
    const qs = new URLSearchParams();
    if (opts?.frame !== undefined) qs.set("frame", String(opts.frame));
    if (opts?.from_idx !== undefined) qs.set("from_idx", String(opts.from_idx));
    if (opts?.to_idx !== undefined) qs.set("to_idx", String(opts.to_idx));
    const tail = qs.toString() ? `?${qs}` : "";
    return request<InferPredictions>(
      `/api/runs/${encodeURIComponent(run_id)}/predictions/${encodeURIComponent(clip_id)}${tail}`
    );
  },
  getPredictionsNoRun: (
    clip_id: string,
    opts?: { frame?: number; from_idx?: number; to_idx?: number }
  ) => {
    const qs = new URLSearchParams();
    if (opts?.frame !== undefined) qs.set("frame", String(opts.frame));
    if (opts?.from_idx !== undefined) qs.set("from_idx", String(opts.from_idx));
    if (opts?.to_idx !== undefined) qs.set("to_idx", String(opts.to_idx));
    const tail = qs.toString() ? `?${qs}` : "";
    return request<InferPredictions>(
      `/api/predictions/${encodeURIComponent(clip_id)}${tail}`
    );
  }
};

export function polygonD(pts: number[], width: number, height: number): string {
  if (pts.length < 4) return "";
  const out: string[] = [];
  for (let i = 0; i < pts.length; i += 2) {
    const x = pts[i] * width;
    const y = pts[i + 1] * height;
    out.push(`${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return out.join(" ") + " Z";
}

export type InferStatus = {
  job_id: string;
  run_id: string;
  clip_id: string;
  status: JobStatus;
  error: string | null;
  done_frames: number;
  total_frames: number;
  predictions_path: string | null;
};

export type InferDetection = {
  class_id: number;
  label: string;
  score: number;
  polygon_norm: number[];
  bbox_norm: [number, number, number, number];
};

export type InferPredictions = {
  run_id: string;
  clip_id: string;
  weights_path: string;
  conf: number;
  iou: number;
  frame_count: number;
  deleted_frames: number[];
  predictions: Record<string, InferDetection[]>;
};

export function inferWebSocket(runId: string, clipId: string, jobId: string): WebSocket {
  return wsUrl(`/ws/infer/${runId}/${clipId}/${jobId}`);
}

export function inferWebSocketNoRun(clipId: string, jobId: string): WebSocket {
  return wsUrl(`/ws/infer/${clipId}/${jobId}`);
}

export function liveInferWebSocket(jobId: string): WebSocket {
  return wsUrl(`/ws/infer/live/${jobId}`);
}

export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";

export const TERMINAL_JOB_STATUSES: ReadonlySet<JobStatus> = new Set([
  "done",
  "error",
  "cancelled"
]);

export const ACTIVE_JOB_STATUSES: ReadonlySet<JobStatus> = new Set(["pending", "running"]);

export type TrainStatus = {
  job_id: string;
  run_id: string;
  status: JobStatus;
  pid: number | null;
  return_code: number | null;
  error: string | null;
  log_tail: string[];
  metrics: Record<string, unknown> | null;
};

function wsUrl(path: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${window.location.host}${path}`);
}

export function trainWebSocket(runId: string, jobId: string): WebSocket {
  return wsUrl(`/ws/train/${runId}/${jobId}`);
}

export type AugmentResponse = {
  run_id: string;
  source_frames: number;
  written_frames: number;
  written_polygons: number;
  copy_paste_inserts: number;
  skipped: { file: string; reason: string }[];
};

export type ExportResponse = {
  run_id: string;
  run_dir: string;
  classes: string[];
  train_frames: number;
  val_frames: number;
  train_polygons: number;
  val_polygons: number;
  skipped_frames: { clip_id: string; frame_idx: number; reason: string }[];
};

export type PropagateResponse = {
  job_id: string;
  clip_id: string;
  start: number;
  end: number;
  total_frames: number;
};

export type PropagateStatus = {
  job_id: string;
  clip_id: string;
  status: JobStatus;
  done_frames: number;
  total_frames: number;
  error: string | null;
  detail: ClipDetail | null;
};

export function propagateWebSocket(clipId: string, jobId: string): WebSocket {
  return wsUrl(`/ws/propagate/${clipId}/${jobId}`);
}

export type CleanupResponse = { detail: ClipDetail; dropped_track_ids: number[] };

export type Sam3ScoreMode = "native" | "per_query";
export type Sam3Config = { score_mode: Sam3ScoreMode };

export type TrackAtResult = {
  track_id: number | null;
  class_id: number | null;
  label: string | null;
};

export function classColor(classId: number, total: number, lightness = 55): string {
  const hue = (classId * 360) / Math.max(total, 1);
  return `hsl(${hue.toFixed(1)} 70% ${lightness}%)`;
}

export type TrackOut = {
  track_id: number;
  class_id: number;
  label: string;
  seeded_from: string;
  frame_indices: number[];
};

export type ClipDetail = {
  clip_id: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  deleted_frames: number[];
  tracks: TrackOut[];
};

export type RunSummary = {
  run_id: string;
  run_dir: string;
  has_data: boolean;
  has_model: boolean;
  has_augment: boolean;
  train_frames: number;
  val_frames: number;
  classes: string[];
  metrics: Record<string, unknown> | null;
  weights_path: string | null;  // Path to best.pt if trained
};

export type ModelInfo = {
  name: string;
  path: string;
  run_id: string | null;
  metrics: Record<string, unknown> | null;
  created_at: number | null;
};

export function labelWebSocket(clipId: string): WebSocket {
  return wsUrl(`/ws/label/${clipId}`);
}

export function maskUrl(clipId: string, trackId: number, frameIdx: number): string {
  return `/api/clips/${clipId}/tracks/${trackId}/masks/${frameIdx}.png`;
}

export function captureWebSocket(clipId: string): WebSocket {
  return wsUrl(`/ws/capture/${clipId}`);
}
