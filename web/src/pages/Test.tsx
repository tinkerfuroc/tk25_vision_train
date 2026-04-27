import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ACTIVE_JOB_STATUSES,
  api,
  classColor,
  inferWebSocket,
  InferDetection,
  JobStatus,
  polygonD,
  TERMINAL_JOB_STATUSES
} from "../api/rest";

export function TestPage() {
  const { runId = "", clipId = "" } = useParams();
  const [weightsPath, setWeightsPath] = useState("yolo_seg_finetuned_best.pt");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [job, setJob] = useState<{
    job_id: string;
    status: JobStatus;
    done: number;
    total: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => () => wsRef.current?.close(), []);

  const clip = useQuery({
    queryKey: ["clip-detail", clipId],
    queryFn: () => api.clipDetail(clipId)
  });
  const ontology = useQuery({ queryKey: ["ontology"], queryFn: api.ontology });
  const predictions = useQuery({
    queryKey: ["predictions", runId, clipId],
    queryFn: () => api.getPredictions(runId, clipId),
    retry: false,
    refetchOnWindowFocus: false
  });

  const startMut = useMutation({
    mutationFn: () => api.startInfer(runId, clipId, { weights_path: weightsPath, conf, iou }),
    onSuccess: (res) => {
      setError(null);
      setJob({ job_id: res.job_id, status: res.status, done: 0, total: res.total_frames });
      wsRef.current?.close();
      const ws = inferWebSocket(res.run_id, res.clip_id, res.job_id);
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        let msg: { event?: string; done?: number; total?: number };
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.event === "ping") return;
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
    },
    onError: (e) => setError((e as Error).message)
  });

  const cancelMut = useMutation({
    mutationFn: () => {
      if (!job) throw new Error("no active job");
      return api.cancelInfer(runId, clipId, job.job_id);
    }
  });

  const [frameIdx, setFrameIdx] = useState(0);
  const dets: InferDetection[] = useMemo(() => {
    const data = predictions.data;
    if (!data) return [];
    return data.predictions[String(frameIdx)] ?? [];
  }, [predictions.data, frameIdx]);

  const totalClasses = ontology.data?.labels.length ?? 1;
  const c = clip.data;
  const frameCount = c?.frame_count ?? 0;
  const isDeleted = c?.deleted_frames.includes(frameIdx) ?? false;

  const active = job && ACTIVE_JOB_STATUSES.has(job.status);

  return (
    <div className="p-4 space-y-3">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-emerald-400 hover:text-emerald-300">← clips</Link>
          <Link to={`/label/${clipId}`} className="text-emerald-400 hover:text-emerald-300">
            ← label
          </Link>
          <h1 className="text-lg font-semibold">Test {runId}/{clipId}</h1>
        </div>
        <div className="flex gap-2 items-center">
          <label className="text-xs text-slate-400 flex items-center gap-1">
            weights
            <input
              type="text"
              value={weightsPath}
              onChange={(e) => setWeightsPath(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs w-72"
            />
          </label>
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
          {active && job ? (
            <button
              onClick={() => cancelMut.mutate()}
              className="rounded bg-rose-800 hover:bg-rose-700 px-3 py-1.5 text-sm"
            >
              Cancel ({job.done}/{job.total})
            </button>
          ) : (
            <button
              onClick={() => startMut.mutate()}
              disabled={startMut.isPending || !weightsPath}
              className="rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 px-3 py-1.5 text-sm"
            >
              {startMut.isPending ? "…" : "Run inference"}
            </button>
          )}
        </div>
      </header>

      {error ? (
        <div className="rounded bg-rose-950 border border-rose-800 p-2 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {job ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-2 text-xs text-slate-300">
          job {job.job_id} — {job.status} — {job.done}/{job.total}
        </div>
      ) : null}

      {c ? (
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
              src={`/api/clips/${clipId}/frames/${frameIdx}`}
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
      ) : null}
    </div>
  );
}
