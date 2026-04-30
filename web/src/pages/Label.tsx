import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ACTIVE_JOB_STATUSES, classColor, JobStatus, labelWebSocket, maskUrl, propagateWebSocket, Sam3ScoreMode, TERMINAL_JOB_STATUSES, TrackOut, trainWebSocket } from "../api/rest";

type Point = [number, number, 0 | 1];

type RefineBody = {
  track_id?: number | null;
  class_id?: number | null;
  label?: string | null;
  points?: Point[];
};

export function LabelPage() {
  const { clipId = "" } = useParams<{ clipId: string }>();
  const qc = useQueryClient();
  const ontology = useQuery({ queryKey: ["ontology"], queryFn: api.ontology });
  const sam3Cfg = useQuery({ queryKey: ["sam3-config"], queryFn: api.getSam3Config });
  const sam3CfgMut = useMutation({
    mutationFn: (mode: Sam3ScoreMode) => api.setSam3Config(mode),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sam3-config"] })
  });
  const detail = useQuery({
    queryKey: ["clip-detail", clipId],
    queryFn: () => api.clipDetail(clipId),
    enabled: !!clipId
  });

  const [frameIdx, setFrameIdx] = useState(0);
  const [points, setPoints] = useState<Point[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const [pickerClass, setPickerClass] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [maskRefresh, setMaskRefresh] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Live state in a ref so the window-keydown handler binds once. Without
  // this, scrubbing with arrow keys would detach + reattach the listener
  // on every keystroke (frameIdx changes → effect re-runs).
  const keyStateRef = useRef({
    selectedTrack,
    points,
    pickerClass,
    frameIdx,
    frameCount: detail.data?.frame_count ?? 1
  });
  keyStateRef.current = {
    selectedTrack,
    points,
    pickerClass,
    frameIdx,
    frameCount: detail.data?.frame_count ?? 1
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      const s = keyStateRef.current;
      if (e.key === "Escape") {
        setPoints([]);
        setSelectedTrack(null);
      } else if (e.key === "r" && s.selectedTrack !== null && s.points.length > 0) {
        refineMut.mutate({ track_id: s.selectedTrack, points: s.points });
      } else if (e.key === "n" && s.points.length > 0) {
        refineMut.mutate({ class_id: s.pickerClass, points: s.points });
      } else if (e.key === "x") {
        frameMut.mutate({ op: "delete", idx: s.frameIdx });
      } else if (e.key === "u") {
        frameMut.mutate({ op: "restore", idx: s.frameIdx });
      } else if (e.key === "p") {
        frameMut.mutate({ op: "prune", idx: s.frameIdx });
      } else if (e.key === "t") {
        startPropagateMut.mutate(s.frameIdx);
      } else if (e.key === "ArrowLeft") {
        setFrameIdx((n) => Math.max(0, n - 1));
      } else if (e.key === "ArrowRight") {
        setFrameIdx((n) => Math.min(s.frameCount - 1, n + 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!clipId) return;
    const ws = labelWebSocket(clipId);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === "refined" || msg.event === "seeded") {
          qc.invalidateQueries({ queryKey: ["clip-detail", clipId] });
          setMaskRefresh((n) => n + 1);
        } else if (msg.event === "error") {
          setError(msg.detail);
        }
      } catch {
        /* ignore */
      }
    };
    return () => {
      ws.onmessage = null;
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    };
  }, [clipId, qc]);

  const seedMut = useMutation({
    mutationFn: () => api.seedClip(clipId),
    onMutate: () => setBusy(true),
    onSuccess: () => {
      setBusy(false);
      setError(null);
      qc.invalidateQueries({ queryKey: ["clip-detail", clipId] });
      setMaskRefresh((n) => n + 1);
    },
    onError: (e) => {
      setBusy(false);
      setError((e as Error).message);
    }
  });

  const refineMut = useMutation({
    mutationFn: (body: RefineBody) => api.refine(clipId, frameIdx, body),
    onMutate: () => setBusy(true),
    onSuccess: () => {
      setBusy(false);
      setError(null);
      setPoints([]);
      qc.invalidateQueries({ queryKey: ["clip-detail", clipId] });
      setMaskRefresh((n) => n + 1);
    },
    onError: (e) => {
      setBusy(false);
      setError((e as Error).message);
    }
  });

  const deleteTrackMut = useMutation({
    mutationFn: (tid: number) => api.deleteTrack(clipId, tid),
    onSuccess: (data) => qc.setQueryData(["clip-detail", clipId], data)
  });

  const frameMut = useMutation({
    mutationFn: ({ op, idx }: { op: "delete" | "restore" | "prune"; idx: number }) => {
      if (op === "delete") return api.deleteFrame(clipId, idx);
      if (op === "restore") return api.restoreFrame(clipId, idx);
      return api.pruneFrames(clipId, idx);
    },
    onSuccess: (data) => qc.setQueryData(["clip-detail", clipId], data),
    onError: (e) => setError((e as Error).message)
  });

  type PropProgress = {
    job_id: string;
    status: "running" | "done" | "cancelled" | "error" | "ready";
    done: number;
    total: number;
    error: string | null;
  };
  const [prop, setProp] = useState<PropProgress | null>(null);
  const propWsRef = useRef<WebSocket | null>(null);

  const startPropagateMut = useMutation({
    mutationFn: (start: number = 0) =>
      api.propagate(clipId, { start, end: null, chunk_size: 25, chunk_overlap: 4, respect_edits: true }),
    onSuccess: (res) => {
      setProp({ job_id: res.job_id, status: "ready", done: 0, total: res.total_frames, error: null });
      const ws = propagateWebSocket(clipId, res.job_id);
      propWsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event === "frame" && typeof msg.done === "number") {
            setProp((cur) => {
              if (!cur) return cur;
              const total = msg.total ?? cur.total;
              if (cur.status === "running" && cur.done === msg.done && cur.total === total) {
                return cur;
              }
              return { ...cur, status: "running", done: msg.done, total };
            });
          } else if (msg.event === "chunk_done") {
            setMaskRefresh((n) => n + 1);
            qc.invalidateQueries({ queryKey: ["clip-detail", clipId] });
          } else if (msg.event === "end") {
            setProp((cur) => (cur ? { ...cur, status: msg.status, error: msg.error ?? null } : cur));
            setMaskRefresh((n) => n + 1);
            qc.invalidateQueries({ queryKey: ["clip-detail", clipId] });
            try { ws.close(); } catch { /* ignore */ }
          }
        } catch { /* ignore */ }
      };
      ws.onerror = () => setProp((cur) => (cur ? { ...cur, status: "error", error: "ws error" } : cur));
    },
    onError: (e) => setError((e as Error).message)
  });

  const cancelPropagateMut = useMutation({
    mutationFn: () => {
      if (!prop) throw new Error("no active propagate");
      return api.cancelPropagate(clipId, prop.job_id);
    }
  });

  const cleanupMut = useMutation({
    mutationFn: () => api.cleanupClip(clipId),
    onSuccess: (res) => {
      qc.setQueryData(["clip-detail", clipId], res.detail);
      setMaskRefresh((n) => n + 1);
      if (res.dropped_track_ids.length === 0) setError("No degenerate tracks to clean.");
      else setError(null);
    },
    onError: (e) => setError((e as Error).message)
  });

  // Persist lastRunId per clip to localStorage so it survives page refresh
  const storageKey = `tk25_lastRunId_${clipId}`;
  const [lastRunId, setLastRunId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storageKey);
    } catch {
      return null;
    }
  });

  // Fetch run details if we have a lastRunId
  const runInfo = useQuery({
    queryKey: ["run", lastRunId],
    queryFn: () => api.getRun(lastRunId!),
    enabled: !!lastRunId,
  });

  // Fetch available models for testing (don't require export first)
  const models = useQuery({
    queryKey: ["models"],
    queryFn: api.models,
  });

  const exportMut = useMutation({
    mutationFn: () => {
      const ts = new Date()
        .toISOString()
        .replace(/[-:.T]/g, "")
        .slice(0, 14); // YYYYMMDDHHMMSS
      const run_id = `${clipId}_${ts}`;
      return api.exportClip(clipId, { run_id, train_ratio: 0.85, per_clip_split: false });
    },
    onSuccess: (res) => {
      setLastRunId(res.run_id);
      try { localStorage.setItem(storageKey, res.run_id); } catch { /* ignore */ }
      setNotice(
        `Exported ${res.train_frames + res.val_frames} frames ` +
          `(${res.train_polygons + res.val_polygons} polygons) → ${res.run_dir}`
      );
      setError(null);
    },
    onError: (e) => setError((e as Error).message)
  });

  const augmentMut = useMutation({
    mutationFn: (run_id: string) => api.augmentRun(run_id, { seed: 0 }),
    onSuccess: (res) => {
      setNotice(
        `Augmented ${res.source_frames} → ${res.written_frames} frames ` +
          `(${res.written_polygons} polygons, ${res.copy_paste_inserts} copy-paste inserts)`
      );
      setError(null);
    },
    onError: (e) => setError((e as Error).message)
  });

  const [train, setTrain] = useState<{ job_id: string; status: JobStatus; log: string[] } | null>(null);
  const trainWsRef = useRef<WebSocket | null>(null);

  // Recover training state from API if we have a lastRunId with a trained model
  useEffect(() => {
    if (lastRunId && runInfo.data?.has_model && !train) {
      // Model exists, show as "done" if not already training
      setTrain({ job_id: "recovered", status: "done", log: [] });
    }
  }, [lastRunId, runInfo.data, train]);

  useEffect(() => {
    return () => {
      trainWsRef.current?.close();
      trainWsRef.current = null;
    };
  }, []);

  const startTrainMut = useMutation({
    mutationFn: (run_id: string) => api.startTrain(run_id, {}),
    onSuccess: (res) => {
      setError(null);
      setNotice(`Training started: job ${res.job_id}`);
      setTrain({ job_id: res.job_id, status: res.status, log: res.log_tail });
      trainWsRef.current?.close();
      const ws = trainWebSocket(res.run_id, res.job_id);
      trainWsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setTrain((cur) => {
            if (!cur || cur.job_id !== res.job_id) return cur;
            if (msg.event === "log" && msg.line) {
              return { ...cur, log: [...cur.log, msg.line].slice(-200) };
            }
            const newStatus: JobStatus | undefined =
              TERMINAL_JOB_STATUSES.has(msg.event as JobStatus) ? (msg.event as JobStatus) : (msg.status as JobStatus | undefined);
            if (newStatus && newStatus !== cur.status) {
              return { ...cur, status: newStatus };
            }
            return cur;
          });
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (trainWsRef.current === ws) trainWsRef.current = null;
      };
    },
    onError: (e) => setError((e as Error).message)
  });

  const cancelTrainMut = useMutation({
    mutationFn: () => {
      if (!train || !lastRunId) throw new Error("no active training job");
      return api.cancelTrain(lastRunId, train.job_id);
    }
  });

  const c = detail.data;
  const totalClasses = ontology.data?.labels.length ?? 1;
  const isDeletedFrame = c?.deleted_frames.includes(frameIdx) ?? false;

  const eventToImageCoords = (
    e: React.MouseEvent<HTMLImageElement>
  ): [number, number] | null => {
    if (!c) return null;
    const target = e.currentTarget as HTMLImageElement;
    const rect = target.getBoundingClientRect();
    const xRatio = c.width / rect.width;
    const yRatio = c.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * xRatio);
    const y = Math.round((e.clientY - rect.top) * yRatio);
    return [x, y];
  };

  const onCanvasClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    const xy = eventToImageCoords(e);
    if (!xy) return;
    const [x, y] = xy;
    const isNegative = e.shiftKey || e.button === 2;
    if (isNegative) {
      setPoints((prev) => [...prev, [x, y, 0]]);
      return;
    }
    try {
      const hit = await api.trackAt(clipId, frameIdx, x, y);
      if (hit?.track_id != null) {
        setSelectedTrack(hit.track_id);
        setPoints([]);
        return;
      }
    } catch {
      /* fall through to point-add */
    }
    setPoints((prev) => [...prev, [x, y, 1]]);
  };

  const onCanvasContext = (e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
    const xy = eventToImageCoords(e);
    if (!xy) return;
    setPoints((prev) => [...prev, [xy[0], xy[1], 0]]);
  };

  const submitRefine = (mode: "edit" | "new") => {
    if (!c || points.length === 0) return;
    const body: RefineBody =
      mode === "edit" && selectedTrack !== null
        ? { points, track_id: selectedTrack }
        : { points, class_id: pickerClass };
    refineMut.mutate(body);
  };

  const visibleTracks: TrackOut[] = c?.tracks ?? [];
  const labels = ontology.data?.labels ?? [];

  if (!clipId) {
    return <div className="p-6 text-rose-400">No clip selected.</div>;
  }

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-center justify-between gap-4">
        <div className="space-y-1">
          <Link to="/" className="text-xs text-slate-400 hover:text-slate-100">
            ← clips
          </Link>
          <h2 className="font-mono text-sm">{clipId}</h2>
          {c ? (
            <div className="text-xs text-slate-500">
              {c.width}×{c.height} · {c.frame_count} fr · {c.tracks.length} track(s)
            </div>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 flex items-center gap-1" title="native = SAM3 default (presence-gated). per_query = ignore presence head, more recall on out-of-distribution prompts.">
            score
            <select
              value={sam3Cfg.data?.score_mode ?? "native"}
              disabled={!sam3Cfg.data || sam3CfgMut.isPending}
              onChange={(e) => sam3CfgMut.mutate(e.target.value as Sam3ScoreMode)}
              className="rounded bg-slate-950 border border-slate-700 px-1.5 py-0.5 text-xs"
            >
              <option value="native">native</option>
              <option value="per_query">per_query</option>
            </select>
          </label>
          <button
            onClick={() => seedMut.mutate()}
            disabled={busy}
            className="rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 px-3 py-1.5 text-sm"
          >
            {busy ? "…" : "Seed first frame (SAM3 text)"}
          </button>
          <button
            onClick={() => cleanupMut.mutate()}
            disabled={!c || cleanupMut.isPending}
            title="Drop persisted tracks whose masks are degenerate (full-frame or edge-touching). Click-edited tracks are spared."
            className="rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 px-3 py-1.5 text-sm"
          >
            {cleanupMut.isPending ? "…" : "Clean stale masks"}
          </button>
          <button
            onClick={() => exportMut.mutate()}
            disabled={!c || c.tracks.length === 0 || exportMut.isPending}
            title="Materialize this clip's labeled frames as a YOLO-seg dataset under data/runs/<run_id>/."
            className="rounded bg-amber-700 hover:bg-amber-600 disabled:opacity-40 px-3 py-1.5 text-sm"
          >
            {exportMut.isPending ? "…" : "Export YOLO-seg"}
          </button>
          <button
            onClick={() => lastRunId && augmentMut.mutate(lastRunId)}
            disabled={!lastRunId || augmentMut.isPending || exportMut.isPending}
            title="Apply Albumentations augmentations to the most recent export."
            className="rounded bg-fuchsia-700 hover:bg-fuchsia-600 disabled:opacity-40 px-3 py-1.5 text-sm"
          >
            {augmentMut.isPending ? "…" : `Augment${lastRunId ? "" : " (export first)"}`}
          </button>
          {train && ACTIVE_JOB_STATUSES.has(train.status) ? (
            <button
              onClick={() => cancelTrainMut.mutate()}
              className="rounded bg-rose-800 hover:bg-rose-700 px-3 py-1.5 text-sm"
            >
              Cancel train
            </button>
          ) : (
            <button
              onClick={() => lastRunId && startTrainMut.mutate(lastRunId)}
              disabled={!lastRunId || startTrainMut.isPending || augmentMut.isPending}
              title="Train YOLO-seg on the most recent export. Streams stdout via WebSocket."
              className="rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 px-3 py-1.5 text-sm"
            >
              {startTrainMut.isPending ? "…" : `Train${lastRunId ? "" : " (export first)"}`}
            </button>
          )}
          {lastRunId && runInfo.data?.has_model ? (
            <Link
              to={`/test/${lastRunId}/${clipId}?weights=${encodeURIComponent(runInfo.data.weights_path || "yolo_seg_finetuned_best.pt")}`}
              className="rounded bg-teal-700 hover:bg-teal-600 px-3 py-1.5 text-sm"
              title="Run inference replay on this clip with the trained weights file."
            >
              Test →
            </Link>
          ) : models.data?.length ? (
            <Link
              to={`/test/${lastRunId || "none"}/${clipId}`}
              className="rounded bg-teal-700 hover:bg-teal-600 px-3 py-1.5 text-sm"
              title="Run inference with any trained model (no export needed for this clip)."
            >
              Test →
            </Link>
          ) : lastRunId ? (
            <Link
              to={`/test/${lastRunId}/${clipId}`}
              className="rounded bg-slate-700 hover:bg-slate-600 px-3 py-1.5 text-sm opacity-50"
              title="No trained models found. Train a model first."
            >
              Test (no models)
            </Link>
          ) : null}
          {prop && (prop.status === "running" || prop.status === "ready") ? (
            <button
              onClick={() => cancelPropagateMut.mutate()}
              className="rounded bg-rose-800 hover:bg-rose-700 px-3 py-1.5 text-sm"
            >
              Cancel ({prop.status === "ready" ? "waiting..." : `${prop.done}/${prop.total}`})
            </button>
          ) : (
            <button
              onClick={() => startPropagateMut.mutate(0)}
              disabled={!c || c.tracks.length === 0 || startPropagateMut.isPending}
              title="Propagate the seeded tracks across all frames using SAM3 per-frame re-detection + IoU matching."
              className="rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40 px-3 py-1.5 text-sm"
            >
              {startPropagateMut.isPending ? "…" : "Propagate"}
            </button>
          )}
        </div>
      </header>

      {error ? (
        <div className="rounded bg-rose-950 border border-rose-800 p-2 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {notice ? (
        <div className="rounded bg-emerald-950 border border-emerald-800 p-2 text-sm text-emerald-200">
          {notice}
        </div>
      ) : null}

      {/* Show run status when we have a lastRunId */}
      {lastRunId && runInfo.data ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-2 text-xs text-slate-300">
          <div className="flex items-center justify-between">
            <span className="font-mono">{lastRunId}</span>
            <span className="text-slate-500">
              {runInfo.data.train_frames + runInfo.data.val_frames} frames
              {runInfo.data.has_augment ? " +aug" : ""}
              {runInfo.data.has_model ? " ✓trained" : ""}
            </span>
          </div>
        </div>
      ) : null}

      {train ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-2 text-xs text-slate-300">
          <div className="flex items-center justify-between mb-1">
            <span>train {train.job_id} — {train.status}</span>
            <span className="text-slate-500">{train.log.length} log lines</span>
          </div>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-slate-400">
            {train.log.slice(-20).join("\n")}
          </pre>
        </div>
      ) : null}

      {prop ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-2 text-xs text-slate-300">
          <div className="flex items-center justify-between mb-1">
            <span>
              propagate {prop.job_id} — {prop.status}
              {prop.error ? `: ${prop.error}` : ""}
            </span>
            <span>{prop.done}/{prop.total}</span>
          </div>
          <div className="h-1.5 w-full bg-slate-800 rounded overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all"
              style={{ width: `${prop.total ? (100 * prop.done) / prop.total : 0}%` }}
            />
          </div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        <div ref={containerRef} className="rounded border border-slate-800 bg-slate-900 p-2">
          {c ? (
            <div className="relative select-none">
              <img
                src={`/api/clips/${clipId}/frames/${frameIdx}`}
                alt={`frame ${frameIdx}`}
                onClick={onCanvasClick}
                onContextMenu={onCanvasContext}
                draggable={false}
                className={`w-full h-auto block cursor-crosshair ${
                  isDeletedFrame ? "opacity-40" : ""
                }`}
              />
              {isDeletedFrame ? (
                <div className="absolute top-2 left-2 rounded bg-rose-900/80 px-2 py-0.5 text-xs text-rose-100 pointer-events-none">
                  deleted
                </div>
              ) : null}
              {(() => {
                const onFrame = visibleTracks.filter((t) => t.frame_indices.includes(frameIdx));
                // Many overlapping translucent masks alpha-soup the canvas.
                // When >6 tracks stack, only render the selected one (if any);
                // user cycles through via the sidebar list.
                const stacked = onFrame.length > 6;
                return onFrame.map((t) => {
                  const isSelected = selectedTrack === t.track_id;
                  const opacity = isSelected
                    ? 0.85
                    : selectedTrack !== null
                      ? 0.0
                      : stacked
                        ? 0.0
                        : 0.22;
                  if (opacity === 0) return null;
                  const url = `${maskUrl(clipId, t.track_id, frameIdx)}?b=${maskRefresh}`;
                  const fill = classColor(t.class_id, totalClasses);
                  return (
                    <div
                      key={t.track_id}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{
                        backgroundColor: fill,
                        WebkitMaskImage: `url(${url})`,
                        maskImage: `url(${url})`,
                        WebkitMaskSize: "100% 100%",
                        maskSize: "100% 100%",
                        WebkitMaskRepeat: "no-repeat",
                        maskRepeat: "no-repeat",
                        WebkitMaskMode: "alpha",
                        maskMode: "alpha",
                        opacity,
                        outline: isSelected
                          ? `2px solid ${classColor(t.class_id, totalClasses, 75)}`
                          : "none"
                      } as React.CSSProperties}
                    />
                  );
                });
              })()}
              {points.map((p, i) => (
                <span
                  key={i}
                  className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full"
                  style={{
                    left: `${(p[0] / c.width) * 100}%`,
                    top: `${(p[1] / c.height) * 100}%`,
                    width: 12,
                    height: 12,
                    background: p[2] === 1 ? "rgb(34,197,94)" : "rgb(239,68,68)",
                    boxShadow: "0 0 0 2px rgba(255,255,255,0.7)"
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="aspect-video rounded grid place-items-center text-slate-500">
              loading clip…
            </div>
          )}
          <div className="mt-2 flex items-center gap-3">
            <button
              onClick={() => setFrameIdx((n) => Math.max(0, n - 1))}
              className="text-xs px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700"
            >
              ←
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(0, (c?.frame_count ?? 1) - 1)}
              value={frameIdx}
              onChange={(e) => setFrameIdx(Number(e.target.value))}
              className="flex-1 accent-emerald-500"
            />
            <button
              onClick={() => setFrameIdx((n) => Math.min((c?.frame_count ?? 1) - 1, n + 1))}
              className="text-xs px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700"
            >
              →
            </button>
            <span className="text-xs text-slate-400 w-16 text-right">
              {frameIdx + 1}/{c?.frame_count ?? 0}
            </span>
          </div>
          {c ? (
            <div className="mt-2 flex items-center gap-2 text-xs">
              {isDeletedFrame ? (
                <>
                  <span className="text-rose-400 font-medium">frame deleted</span>
                  <button
                    disabled={frameMut.isPending}
                    onClick={() => frameMut.mutate({ op: "restore", idx: frameIdx })}
                    className="rounded bg-emerald-800 hover:bg-emerald-700 disabled:opacity-30 px-2 py-0.5"
                  >
                    Restore (u)
                  </button>
                </>
              ) : (
                <button
                  disabled={frameMut.isPending}
                  onClick={() => frameMut.mutate({ op: "delete", idx: frameIdx })}
                  className="rounded bg-rose-800 hover:bg-rose-700 disabled:opacity-30 px-2 py-0.5"
                >
                  Delete frame (x)
                </button>
              )}
              <button
                disabled={frameMut.isPending || frameIdx >= c.frame_count - 1}
                onClick={() => frameMut.mutate({ op: "prune", idx: frameIdx + 1 })}
                title="Mark every frame after this one as deleted."
                className="rounded bg-rose-900 hover:bg-rose-800 disabled:opacity-30 px-2 py-0.5"
              >
                Prune after (p)
              </button>
              <button
                disabled={
                  startPropagateMut.isPending || c.tracks.length === 0 || (prop?.status === "running")
                }
                onClick={() => startPropagateMut.mutate(frameIdx)}
                title="Re-propagate from this frame to the end."
                className="rounded bg-indigo-800 hover:bg-indigo-700 disabled:opacity-30 px-2 py-0.5"
              >
                Re-propagate from here (t)
              </button>
              <span className="text-slate-500 ml-auto">{c.deleted_frames.length} deleted</span>
            </div>
          ) : null}
          <div className="mt-1 text-xs text-slate-500">
            click on mask = select · click on bg = positive prompt · shift/right-click = negative · r = refine selected · n = new track · esc = clear · ←/→ scrub · x delete · u restore · p prune · t propagate-from-here · {points.length} point(s){selectedTrack !== null ? ` · selected #${selectedTrack}` : ""}
          </div>
        </div>

        <aside className="space-y-3">
          <section className="rounded border border-slate-800 bg-slate-900 p-3">
            <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-2">Refine</h3>
            <div className="space-y-2">
              <button
                disabled={busy || points.length === 0 || selectedTrack === null}
                onClick={() => submitRefine("edit")}
                className="w-full rounded bg-amber-700 hover:bg-amber-600 disabled:opacity-30 px-3 py-1.5 text-sm"
              >
                Refine selected track ({points.length} pts)
              </button>
              <div className="flex gap-2">
                <select
                  value={pickerClass}
                  onChange={(e) => setPickerClass(Number(e.target.value))}
                  className="flex-1 rounded bg-slate-950 border border-slate-700 px-2 py-1 text-sm"
                >
                  {labels.map((l, i) => (
                    <option key={l} value={i}>
                      {l}
                    </option>
                  ))}
                </select>
                <button
                  disabled={busy || points.length === 0}
                  onClick={() => submitRefine("new")}
                  className="rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-30 px-3 py-1.5 text-sm"
                >
                  New track
                </button>
              </div>
              <button
                onClick={() => setPoints([])}
                className="w-full text-xs text-slate-400 hover:text-slate-100"
              >
                clear points
              </button>
            </div>
          </section>

          <section className="rounded border border-slate-800 bg-slate-900 p-3">
            <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-2">Tracks</h3>
            {visibleTracks.length === 0 ? (
              <div className="text-xs text-slate-500">No tracks yet. Hit Seed or click + New track.</div>
            ) : (
              <ul className="space-y-1 text-sm">
                {visibleTracks.map((t) => (
                  <li
                    key={t.track_id}
                    onClick={() =>
                      setSelectedTrack((cur) => (cur === t.track_id ? null : t.track_id))
                    }
                    className={`rounded px-2 py-1 cursor-pointer flex items-center justify-between ${
                      selectedTrack === t.track_id
                        ? "bg-slate-700"
                        : "bg-slate-950 hover:bg-slate-800"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <span
                        className="inline-block w-3 h-3 rounded-sm"
                        style={{ background: classColor(t.class_id, totalClasses) }}
                        title={t.label}
                      />
                      <span className="text-slate-300">{t.label}</span>
                      <span className="text-xs text-slate-500">
                        #{t.track_id} · {t.frame_indices.length} fr · {t.seeded_from}
                      </span>
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteTrackMut.mutate(t.track_id);
                      }}
                      className="text-xs text-slate-500 hover:text-rose-400"
                    >
                      delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
