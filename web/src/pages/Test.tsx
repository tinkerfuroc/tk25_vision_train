import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ACTIVE_JOB_STATUSES,
  api,
  classColor,
  inferWebSocket,
  inferWebSocketNoRun,
  InferDetection,
  JobStatus,
  liveInferWebSocket,
  LiveFrameEvent,
  polygonD,
  TERMINAL_JOB_STATUSES
} from "../api/rest";

export function TestPage() {
  const { runId = "", clipId = "" } = useParams();
  const [searchParams] = useSearchParams();

  // Fetch available clips, models, and ontology
  const clips = useQuery({ queryKey: ["clips"], queryFn: api.clips });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const ontology = useQuery({ queryKey: ["ontology"], queryFn: api.ontology });

  // Selected clip and model
  const urlWeights = searchParams.get("weights");
  const [selectedClipId, setSelectedClipId] = useState(clipId || "");
  const [weightsPath, setWeightsPath] = useState(urlWeights || "");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [mode, setMode] = useState<"clip" | "live">("clip");

  // Clip inference job state
  const [job, setJob] = useState<{
    job_id: string;
    status: JobStatus;
    done: number;
    total: number;
  } | null>(null);
  const [inferError, setInferError] = useState<string | null>(null);
  const inferWsRef = useRef<WebSocket | null>(null);

  // Live inference state
  const [liveJob, setLiveJob] = useState<{
    job_id: string;
    status: JobStatus;
    error?: string;
  } | null>(null);
  const [liveFrame, setLiveFrame] = useState<LiveFrameEvent | null>(null);
  const [liveFrameUrl, setLiveFrameUrl] = useState<string | null>(null);
  const liveWsRef = useRef<WebSocket | null>(null);

  // Frame navigation for clip test
  const [frameIdx, setFrameIdx] = useState(0);

  // Auto-select first clip and model
  useEffect(() => {
    if (!selectedClipId && clips.data?.length) {
      setSelectedClipId(clips.data[0].clip_id);
    }
  }, [selectedClipId, clips.data]);

  useEffect(() => {
    if (!weightsPath && models.data?.length) {
      setWeightsPath(models.data[0].path);
    }
  }, [weightsPath, models.data]);

  // Fetch clip detail for selected clip
  const clipDetail = useQuery({
    queryKey: ["clip-detail", selectedClipId],
    queryFn: () => api.clipDetail(selectedClipId),
    enabled: !!selectedClipId && mode === "clip",
  });

  // Fetch predictions - use runId if available, otherwise use temp_infer
  const effectiveRunId = runId || "temp_infer";
  const predictions = useQuery({
    queryKey: ["predictions", effectiveRunId, selectedClipId],
    queryFn: () => runId
      ? api.getPredictions(runId, selectedClipId)
      : api.getPredictionsNoRun(selectedClipId),
    retry: false,
    refetchOnWindowFocus: false,
    enabled: !!selectedClipId && mode === "clip",
  });

  // Cleanup WebSocket on unmount
  useEffect(() => () => {
    inferWsRef.current?.close();
    liveWsRef.current?.close();
  }, []);

  // Decode base64 frame to blob URL for display
  useEffect(() => {
    if (liveFrame?.frame_b64) {
      const bytes = Uint8Array.from(atob(liveFrame.frame_b64), c => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: "image/jpeg" });
      setLiveFrameUrl(URL.createObjectURL(blob));
    }
  }, [liveFrame?.frame_b64]);

  // Clip inference mutation
  const startMut = useMutation({
    mutationFn: () => {
      if (!selectedClipId || !weightsPath) throw new Error("clip and weights required");
      console.log("Starting inference:", { runId, selectedClipId, weightsPath, conf, iou });
      setInferError(null);
      if (runId) {
        return api.startInfer(runId, selectedClipId, { weights_path: weightsPath, conf, iou });
      } else {
        return api.startInferNoRun(selectedClipId, { weights_path: weightsPath, conf, iou });
      }
    },
    onSuccess: (res) => {
      console.log("Inference started:", res);
      setInferError(null);
      setJob({ job_id: res.job_id, status: res.status, done: 0, total: res.total_frames });
      inferWsRef.current?.close();
      const wsUrl = runId
        ? `/ws/infer/${runId}/${selectedClipId}/${res.job_id}`
        : `/ws/infer/${selectedClipId}/${res.job_id}`;
      console.log("WebSocket URL:", wsUrl);
      const ws = runId
        ? inferWebSocket(runId, selectedClipId, res.job_id)
        : inferWebSocketNoRun(selectedClipId, res.job_id);
      inferWsRef.current = ws;
      ws.onmessage = (ev) => {
        console.log("WS message:", ev.data);
        let msg: { event?: string; done?: number; total?: number; detail?: string };
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.event === "ping") return;
        if (msg.event === "error") {
          setInferError(msg.detail || "WebSocket error");
          setJob((cur) => cur ? { ...cur, status: "error" } : null);
          return;
        }
        setJob((cur) => {
          if (!cur || cur.job_id !== res.job_id) return cur;
          if (msg.event === "frame") {
            return { ...cur, done: msg.done ?? cur.done, total: msg.total ?? cur.total };
          }
          if (msg.event && TERMINAL_JOB_STATUSES.has(msg.event as JobStatus)) {
            if (msg.event === "done") predictions.refetch();
            return { ...cur, status: msg.event as JobStatus };
          }
          return cur;
        });
      };
      ws.onerror = (ev) => {
        console.error("WebSocket error:", ev);
        setInferError("WebSocket connection failed");
      };
    },
    onError: (e) => {
      console.error("Inference error:", e);
      setInferError((e as Error).message);
    },
  });

  const cancelMut = useMutation({
    mutationFn: () => {
      if (!job) throw new Error("no active job");
      if (runId) {
        return api.cancelInfer(runId, selectedClipId, job.job_id);
      } else {
        // For temp_infer, use the run_id from the job response
        return api.cancelInfer("temp_infer", selectedClipId, job.job_id);
      }
    },
  });

  // Live inference mutation
  const startLiveMut = useMutation({
    mutationFn: () => api.startLiveInfer({ weights_path: weightsPath, conf, iou }),
    onSuccess: (res) => {
      setLiveJob({ job_id: res.job_id, status: res.status, error: res.error ?? undefined });
      setLiveFrame(null);
      setLiveFrameUrl(null);
      liveWsRef.current?.close();
      const ws = liveInferWebSocket(res.job_id);
      liveWsRef.current = ws;
      ws.onmessage = (ev) => {
        let msg: LiveFrameEvent & { event?: string; status?: string; detail?: string };
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.event === "ping") return;
        if (msg.event === "error") {
          setLiveJob((cur) => cur ? { ...cur, status: "error", error: msg.detail || "Unknown error" } : cur);
          return;
        }
        if (msg.event === "frame") {
          setLiveFrame(msg);
        }
        if (msg.event && TERMINAL_JOB_STATUSES.has(msg.event as JobStatus)) {
          setLiveJob((cur) => cur ? { ...cur, status: msg.event as JobStatus } : cur);
        }
      };
    },
    onError: (e) => {
      setLiveJob({ job_id: "", status: "error", error: (e as Error).message });
    },
  });

  const cancelLiveMut = useMutation({
    mutationFn: () => {
      if (!liveJob) throw new Error("no active live job");
      return api.cancelLiveInfer(liveJob.job_id);
    },
  });

  // Detections for current frame
  const dets: InferDetection[] = useMemo(() => {
    if (mode === "live" && liveFrame) {
      return liveFrame.detections;
    }
    const data = predictions.data;
    if (!data) return [];
    return data.predictions[String(frameIdx)] ?? [];
  }, [predictions.data, frameIdx, mode, liveFrame]);

  const totalClasses = ontology.data?.labels.length ?? 1;
  const c = clipDetail.data;
  const frameCount = c?.frame_count ?? 0;
  const isDeleted = c?.deleted_frames.includes(frameIdx) ?? false;

  const inferActive = job && ACTIVE_JOB_STATUSES.has(job.status);
  const liveActive = liveJob && ACTIVE_JOB_STATUSES.has(liveJob.status);

  const liveWidth = liveFrame?.width ?? 640;
  const liveHeight = liveFrame?.height ?? 480;

  return (
    <div className="p-4 space-y-3">
      <header className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-emerald-400 hover:text-emerald-300">← clips</Link>
          {selectedClipId ? (
            <Link to={`/label/${selectedClipId}`} className="text-emerald-400 hover:text-emerald-300">
              ← label
            </Link>
          ) : null}
          <h1 className="text-lg font-semibold">Test {runId ? `${runId}/` : ""}{selectedClipId}</h1>
        </div>
        <div className="flex gap-2 items-center flex-wrap">
          {/* Mode selector */}
          <label className="text-xs text-slate-400 flex items-center gap-1">
            mode
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as "clip" | "live")}
              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs"
            >
              <option value="clip">Clip test</option>
              <option value="live">Live camera</option>
            </select>
          </label>

          {/* Clip selector (only for clip mode) */}
          {mode === "clip" && (
            <label className="text-xs text-slate-400 flex items-center gap-1">
              clip
              <select
                value={selectedClipId}
                onChange={(e) => setSelectedClipId(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs w-48"
              >
                {clips.isLoading ? (
                  <option value="">Loading...</option>
                ) : clips.data?.length ? (
                  clips.data.map((c) => (
                    <option key={c.clip_id} value={c.clip_id}>{c.clip_id}</option>
                  ))
                ) : (
                  <option value="">No clips</option>
                )}
              </select>
            </label>
          )}

          {/* Model selector */}
          <label className="text-xs text-slate-400 flex items-center gap-1">
            model
            <select
              value={weightsPath}
              onChange={(e) => setWeightsPath(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs w-64"
            >
              {models.isLoading ? (
                <option value="">Loading...</option>
              ) : models.data?.length ? (
                models.data.map((m) => {
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const map50 = (m.metrics as any)?.metrics?.mAP50;
                  const label = map50 != null ? `${m.name} (mAP50: ${map50.toFixed(2)})` : m.name;
                  return <option key={m.path} value={m.path}>{label}</option>;
                })
              ) : (
                <option value="">No trained models found</option>
              )}
            </select>
          </label>

          {/* Conf and IoU */}
          <label className="text-xs text-slate-400 flex items-center gap-1">
            conf
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={conf}
              onChange={(e) => setConf(Number(e.target.value))}
              className="bg-slate-900 border border-slate-700 rounded px-1 py-1 text-xs w-16"
            />
          </label>
          <label className="text-xs text-slate-400 flex items-center gap-1">
            iou
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={iou}
              onChange={(e) => setIou(Number(e.target.value))}
              className="bg-slate-900 border border-slate-700 rounded px-1 py-1 text-xs w-16"
            />
          </label>

          {/* Run/Cancel buttons */}
          {mode === "clip" ? (
            inferActive && job ? (
              <button
                onClick={() => cancelMut.mutate()}
                className="rounded bg-rose-800 hover:bg-rose-700 px-3 py-1.5 text-sm"
              >
                Cancel ({job.done}/{job.total})
              </button>
            ) : (
              <button
                onClick={() => startMut.mutate()}
                disabled={startMut.isPending || !weightsPath || !selectedClipId}
                className="rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 px-3 py-1.5 text-sm"
              >
                {startMut.isPending ? "…" : "Run on clip"}
              </button>
            )
          ) : (
            liveActive && liveJob ? (
              <button
                onClick={() => cancelLiveMut.mutate()}
                className="rounded bg-rose-800 hover:bg-rose-700 px-3 py-1.5 text-sm"
              >
                Stop live
              </button>
            ) : (
              <button
                onClick={() => startLiveMut.mutate()}
                disabled={startLiveMut.isPending || !weightsPath}
                className="rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-40 px-3 py-1.5 text-sm"
              >
                {startLiveMut.isPending ? "…" : "Live test"}
              </button>
            )
          )}
          {!weightsPath && models.data?.length === 0 ? (
            <span className="text-xs text-amber-400">Need trained model → Datasets page</span>
          ) : null}
        </div>
      </header>

      {/* Job status */}
      {inferError ? (
        <div className="rounded bg-rose-950 border border-rose-800 p-2 text-xs text-rose-200">
          Error: {inferError}
        </div>
      ) : null}
      {job ? (
        <div className={`rounded p-2 text-xs ${job.status === "error" ? "bg-rose-950 border border-rose-800 text-rose-200" : "bg-slate-900 border border-slate-800 text-slate-300"}`}>
          clip infer {job.job_id} — {job.status} — {job.done}/{job.total}
        </div>
      ) : null}

      {liveJob ? (
        <div className={`rounded p-2 text-xs ${liveJob.error ? "bg-rose-950 border border-rose-800 text-rose-200" : "bg-slate-900 border border-slate-800 text-slate-300"}`}>
          live infer {liveJob.job_id} — {liveJob.status} — {liveFrame?.frame_idx ?? 0} frames
          {liveJob.error ? <div className="mt-1 text-rose-300">{liveJob.error}</div> : null}
        </div>
      ) : null}

      {/* Frame display */}
      {mode === "clip" && c ? (
        <div className="space-y-2">
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <button
              onClick={() => setFrameIdx((i) => Math.max(0, i - 1))}
              className="rounded bg-slate-800 hover:bg-slate-700 px-2 py-1"
            >
              ←
            </button>
            <span>frame {frameIdx} / {frameCount - 1}</span>
            <input
              type="range"
              min={0}
              max={Math.max(0, frameCount - 1)}
              value={frameIdx}
              onChange={(e) => setFrameIdx(Number(e.target.value))}
              className="flex-1"
            />
            <button
              onClick={() => setFrameIdx((i) => Math.min(frameCount - 1, i + 1))}
              className="rounded bg-slate-800 hover:bg-slate-700 px-2 py-1"
            >
              →
            </button>
            {isDeleted ? <span className="text-amber-300">(deleted)</span> : null}
            <span className="text-slate-500">{dets.length} detections</span>
          </div>
          <div
            className="relative inline-block border border-slate-800"
            style={{ width: c.width, height: c.height }}
          >
            <img
              src={`/api/clips/${selectedClipId}/frames/${frameIdx}`}
              alt={`frame ${frameIdx}`}
              width={c.width}
              height={c.height}
              draggable={false}
              className={isDeleted ? "opacity-40" : ""}
            />
            <svg
              className="absolute inset-0 pointer-events-none"
              viewBox={`0 0 ${c.width} ${c.height}`}
              width={c.width}
              height={c.height}
            >
              {dets.map((d, i) => {
                const color = classColor(d.class_id, totalClasses);
                return (
                  <g key={i}>
                    <path d={polygonD(d.polygon_norm, c.width, c.height)} fill={color} fillOpacity={0.25} stroke={color} strokeWidth={1.5} />
                    <text
                      x={d.bbox_norm[0] * c.width + 2}
                      y={Math.max(10, d.bbox_norm[1] * c.height - 2)}
                      fontSize="10"
                      fill={color}
                      stroke="black"
                      strokeWidth="0.4"
                    >
                      {d.label} {d.score.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      ) : mode === "live" && liveFrame ? (
        <div className="space-y-2">
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span>live frame {liveFrame.frame_idx}</span>
            <span className="text-slate-500">{liveFrame.detections.length} detections</span>
            <span className="text-slate-500">{liveWidth}×{liveHeight}</span>
          </div>
          <div
            className="relative inline-block border border-slate-800 bg-slate-900"
            style={{ width: liveWidth, height: liveHeight }}
          >
            {liveFrameUrl ? (
              <img
                src={liveFrameUrl}
                alt={`live frame ${liveFrame.frame_idx}`}
                width={liveWidth}
                height={liveHeight}
                draggable={false}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
                Waiting for frame...
              </div>
            )}
            <svg
              className="absolute inset-0 pointer-events-none"
              viewBox={`0 0 ${liveWidth} ${liveHeight}`}
              width={liveWidth}
              height={liveHeight}
            >
              {liveFrame.detections.map((d, i) => {
                const color = classColor(d.class_id, totalClasses);
                return (
                  <g key={i}>
                    <path d={polygonD(d.polygon_norm, liveWidth, liveHeight)} fill={color} fillOpacity={0.25} stroke={color} strokeWidth={1.5} />
                    <text
                      x={d.bbox_norm[0] * liveWidth + 2}
                      y={Math.max(10, d.bbox_norm[1] * liveHeight - 2)}
                      fontSize="10"
                      fill={color}
                      stroke="black"
                      strokeWidth="0.4"
                    >
                      {d.label} {d.score.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      ) : mode === "live" ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-4 text-center text-slate-400 text-sm">
          Click "Live test" to start real-time inference on the connected camera
        </div>
      ) : (
        <div className="rounded bg-slate-900 border border-slate-800 p-4 text-center text-slate-400 text-sm">
          Select a clip and model, then click "Run on clip" to test
          {!runId && " (predictions stored in temp_infer run)"}
        </div>
      )}
    </div>
  );
}