import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, JobStatus, trainWebSocket } from "../api/rest";

export function DatasetsPage() {
  // Fetch all clips, runs, and models
  const clips = useQuery({ queryKey: ["clips"], queryFn: api.clips });
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });

  // Multi-clip export state
  const [selectedClips, setSelectedClips] = useState<Set<string>>(new Set());
  const [runId, setRunId] = useState("");
  const [trainRatio, setTrainRatio] = useState(0.85);
  const [perClipSplit, setPerClipSplit] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Training state per run
  const [trainingRun, setTrainingRun] = useState<string | null>(null);
  const [trainJob, setTrainJob] = useState<{ job_id: string; status: JobStatus; log: string[] } | null>(null);
  const trainWsRef = useRef<WebSocket | null>(null);

  // Cleanup WebSocket on unmount
  useEffect(() => () => {
    trainWsRef.current?.close();
  }, []);

  // Generate default run_id
  const defaultRunId = () => {
    const ts = new Date().toISOString().replace(/[-:.T]/g, "").slice(0, 14);
    return `dataset_${ts}`;
  };

  // Export + Train combined mutation (for multi-clip training)
  const exportAndTrainMut = useMutation({
    mutationFn: async () => {
      const id = runId || defaultRunId();
      const clipIds = selectedClips.size > 0 ? Array.from(selectedClips) : undefined;
      // First export
      const exportRes = await api.exportRun({
        run_id: id,
        clip_ids: clipIds,
        train_ratio: trainRatio,
        per_clip_split: perClipSplit,
      });
      // Then start training on that run
      const trainRes = await api.startTrain(id, {});
      return { exportRes, trainRes, run_id: id };
    },
    onSuccess: ({ exportRes, trainRes, run_id }) => {
      setNotice(
        `Exported ${exportRes.train_frames + exportRes.val_frames} frames from ` +
        `${selectedClips.size || clips.data?.length || 0} clips, training started`
      );
      setError(null);
      runs.refetch();
      setTrainingRun(run_id);
      setTrainJob({ job_id: trainRes.job_id, status: trainRes.status, log: trainRes.log_tail });
      trainWsRef.current?.close();
      const ws = trainWebSocket(run_id, trainRes.job_id);
      trainWsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setTrainJob((cur) => {
            if (!cur || cur.job_id !== trainRes.job_id) return cur;
            if (msg.event === "log" && msg.line) {
              return { ...cur, log: [...cur.log, msg.line].slice(-200) };
            }
            if (msg.event && ["done", "error", "cancelled"].includes(msg.event)) {
              runs.refetch();
              models.refetch();
              return { ...cur, status: msg.event as JobStatus };
            }
            return cur;
          });
        } catch {
          /* ignore */
        }
      };
    },
    onError: (e) => setError((e as Error).message),
  });

  // Export mutation
  const exportMut = useMutation({
    mutationFn: () => {
      const id = runId || defaultRunId();
      const clipIds = selectedClips.size > 0 ? Array.from(selectedClips) : undefined;
      return api.exportRun({
        run_id: id,
        clip_ids: clipIds,
        train_ratio: trainRatio,
        per_clip_split: perClipSplit,
      });
    },
    onSuccess: (res) => {
      setNotice(
        `Exported ${res.train_frames + res.val_frames} frames ` +
        `(${res.train_polygons + res.val_polygons} polygons) from ` +
        `${selectedClips.size || clips.data?.length || 0} clips → ${res.run_dir}`
      );
      setError(null);
      runs.refetch();
    },
    onError: (e) => setError((e as Error).message),
  });

  // Augment mutation
  const augmentMut = useMutation({
    mutationFn: (runId: string) => api.augmentRun(runId, { seed: 0 }),
    onSuccess: (res) => {
      setNotice(`Augmented ${res.source_frames} → ${res.written_frames} frames (${res.written_polygons} polygons)`);
      setError(null);
      runs.refetch();
    },
    onError: (e) => setError((e as Error).message),
  });

  // Start training mutation
  const startTrainMut = useMutation({
    mutationFn: (runId: string) => api.startTrain(runId, {}),
    onSuccess: (res) => {
      setTrainingRun(res.run_id);
      setTrainJob({ job_id: res.job_id, status: res.status, log: res.log_tail });
      setError(null);
      setNotice(`Training started: ${res.job_id}`);
      trainWsRef.current?.close();
      const ws = trainWebSocket(res.run_id, res.job_id);
      trainWsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          setTrainJob((cur) => {
            if (!cur || cur.job_id !== res.job_id) return cur;
            if (msg.event === "log" && msg.line) {
              return { ...cur, log: [...cur.log, msg.line].slice(-200) };
            }
            if (msg.event && ["done", "error", "cancelled"].includes(msg.event)) {
              runs.refetch();
              models.refetch();
              return { ...cur, status: msg.event as JobStatus };
            }
            return cur;
          });
        } catch {
          /* ignore */
        }
      };
    },
    onError: (e) => setError((e as Error).message),
  });

  // Cancel training mutation
  const cancelTrainMut = useMutation({
    mutationFn: () => {
      if (!trainJob || !trainingRun) throw new Error("no active training");
      return api.cancelTrain(trainingRun, trainJob.job_id);
    },
    onSuccess: () => {
      setNotice("Training cancelled");
    },
  });

  const toggleClip = (clipId: string) => {
    const next = new Set(selectedClips);
    if (next.has(clipId)) {
      next.delete(clipId);
    } else {
      next.add(clipId);
    }
    setSelectedClips(next);
  };

  const selectAll = () => {
    if (clips.data) {
      setSelectedClips(new Set(clips.data.map(c => c.clip_id)));
    }
  };

  const deselectAll = () => {
    setSelectedClips(new Set());
  };

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Datasets & Training</h1>
        <Link to="/" className="text-emerald-400 hover:text-emerald-300">← clips</Link>
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

      {/* Training job status */}
      {trainJob ? (
        <div className="rounded bg-slate-900 border border-slate-800 p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium">
              Training {trainingRun} — {trainJob.status}
            </span>
            {trainJob.status === "running" ? (
              <button
                onClick={() => cancelTrainMut.mutate()}
                className="rounded bg-rose-800 hover:bg-rose-700 px-2 py-1 text-xs"
              >
                Cancel
              </button>
            ) : null}
          </div>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-slate-400">
            {trainJob.log.slice(-30).join("\n")}
          </pre>
        </div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Clip selection */}
        <div className="rounded border border-slate-800 bg-slate-900 p-3">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-medium">Clips ({clips.data?.length || 0})</h2>
            <div className="flex gap-2">
              <button
                onClick={selectAll}
                className="text-xs text-slate-400 hover:text-slate-200"
              >
                Select all
              </button>
              <button
                onClick={deselectAll}
                className="text-xs text-slate-400 hover:text-slate-200"
              >
                Deselect all
              </button>
            </div>
          </div>
          <div className="max-h-48 overflow-auto space-y-1">
            {clips.isLoading ? (
              <div className="text-slate-500 text-sm">Loading...</div>
            ) : clips.data?.length ? (
              clips.data.map((c) => (
                <label
                  key={c.clip_id}
                  className="flex items-center gap-2 p-1 rounded hover:bg-slate-800 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selectedClips.has(c.clip_id)}
                    onChange={() => toggleClip(c.clip_id)}
                    className="rounded"
                  />
                  <span className="text-sm flex-1">{c.clip_id}</span>
                  <span className="text-xs text-slate-500">{c.frame_count} fr</span>
                </label>
              ))
            ) : (
              <div className="text-slate-500 text-sm">No clips</div>
            )}
          </div>
          <div className="mt-2 text-xs text-slate-400">
            {selectedClips.size > 0
              ? `${selectedClips.size} clips selected`
              : "All clips will be exported (none selected)"}
          </div>
          {/* Direct multi-clip train button */}
          {selectedClips.size > 0 && !trainJob?.status ? (
            <div className="mt-3 pt-2 border-t border-slate-700">
              <p className="text-xs text-slate-400 mb-2">
                Quick train: export selected clips as one dataset and train immediately
              </p>
              <button
                onClick={() => exportAndTrainMut.mutate()}
                disabled={exportAndTrainMut.isPending}
                className="w-full rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-40 px-3 py-2 text-sm font-medium"
              >
                {exportAndTrainMut.isPending ? "Exporting & Training..." : `Train ${selectedClips.size} Clips →`}
              </button>
            </div>
          ) : null}
        </div>

        {/* Export controls */}
        <div className="rounded border border-slate-800 bg-slate-900 p-3 space-y-3">
          <h2 className="font-medium">Export New Dataset</h2>

          <label className="block text-xs text-slate-400">
            Run ID (optional)
            <input
              type="text"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              placeholder={defaultRunId()}
              className="mt-1 block w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm"
            />
          </label>

          <label className="block text-xs text-slate-400">
            Train ratio: {trainRatio}
            <input
              type="range"
              min={0.5}
              max={0.95}
              step={0.05}
              value={trainRatio}
              onChange={(e) => setTrainRatio(Number(e.target.value))}
              className="mt-1 block w-full"
            />
          </label>

          <label className="flex items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={perClipSplit}
              onChange={(e) => setPerClipSplit(e.target.checked)}
              className="rounded"
            />
            Per-clip split (whole clips → train or val)
          </label>

          <button
            onClick={() => exportMut.mutate()}
            disabled={exportMut.isPending}
            className="w-full rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 px-3 py-2 text-sm font-medium"
          >
            {exportMut.isPending ? "Exporting..." : "Export Dataset"}
          </button>
        </div>
      </div>

      {/* Existing runs - with full training actions */}
      <div className="rounded border border-slate-800 bg-slate-900 p-3">
        <h2 className="font-medium mb-2">Datasets ({runs.data?.length || 0})</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-800">
                <th className="p-2">Run ID</th>
                <th className="p-2">Frames</th>
                <th className="p-2">Status</th>
                <th className="p-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.isLoading ? (
                <tr><td colSpan={4} className="p-2 text-slate-500">Loading...</td></tr>
              ) : runs.data?.length ? (
                runs.data.map((r) => {
                  const isTrainingThis = trainingRun === r.run_id && trainJob?.status === "running";
                  const canAugment = r.has_data && !r.has_augment;
                  const canTrain = r.has_data && !r.has_model;
                  const canTest = r.has_model;

                  return (
                    <tr key={r.run_id} className="border-b border-slate-800">
                      <td className="p-2 font-mono text-xs">{r.run_id}</td>
                      <td className="p-2">
                        {r.train_frames + r.val_frames}
                        {r.has_augment ? " (aug)" : ""}
                      </td>
                      <td className="p-2">
                        {r.has_model ? (
                          <span className="text-emerald-400">trained</span>
                        ) : r.has_augment ? (
                          <span className="text-fuchsia-400">augmented</span>
                        ) : r.has_data ? (
                          <span className="text-cyan-400">exported</span>
                        ) : (
                          <span className="text-slate-500">-</span>
                        )}
                      </td>
                      <td className="p-2">
                        <div className="flex gap-2 flex-wrap">
                          {canAugment ? (
                            <button
                              onClick={() => augmentMut.mutate(r.run_id)}
                              disabled={augmentMut.isPending}
                              className="rounded bg-fuchsia-700 hover:bg-fuchsia-600 disabled:opacity-40 px-2 py-0.5 text-xs"
                            >
                              Augment
                            </button>
                          ) : null}
                          {canTrain ? (
                            <button
                              onClick={() => startTrainMut.mutate(r.run_id)}
                              disabled={startTrainMut.isPending || isTrainingThis}
                              className="rounded bg-teal-700 hover:bg-teal-600 disabled:opacity-40 px-2 py-0.5 text-xs"
                            >
                              Train
                            </button>
                          ) : null}
                          {canTest ? (
                            <Link
                              to={`/test/${r.run_id}`}
                              className="rounded bg-cyan-700 hover:bg-cyan-600 px-2 py-0.5 text-xs"
                            >
                              Test →
                            </Link>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr><td colSpan={4} className="p-2 text-slate-500">No datasets yet. Export clips above to create one.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Trained models */}
      <div className="rounded border border-slate-800 bg-slate-900 p-3">
        <h2 className="font-medium mb-2">Trained Models ({models.data?.length || 0})</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-800">
                <th className="p-2">Name</th>
                <th className="p-2">Path</th>
                <th className="p-2">mAP50</th>
                <th className="p-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.isLoading ? (
                <tr><td colSpan={4} className="p-2 text-slate-500">Loading...</td></tr>
              ) : models.data?.length ? (
                models.data.map((m) => {
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const map50 = (m.metrics as any)?.metrics?.mAP50;
                  return (
                    <tr key={m.path} className="border-b border-slate-800">
                      <td className="p-2">{m.name}</td>
                      <td className="p-2 font-mono text-xs text-slate-400">{m.path}</td>
                      <td className="p-2">{map50 != null ? map50.toFixed(3) : "-"}</td>
                      <td className="p-2">
                        <Link
                          to={`/test?weights=${encodeURIComponent(m.path)}`}
                          className="text-cyan-400 hover:text-cyan-300 text-xs"
                        >
                          Test →
                        </Link>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr><td colSpan={4} className="p-2 text-slate-500">No trained models. Train a dataset above.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
