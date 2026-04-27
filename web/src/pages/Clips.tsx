import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, captureWebSocket } from "../api/rest";

export function ClipsPage() {
  const qc = useQueryClient();
  const clips = useQuery({ queryKey: ["clips"], queryFn: api.clips, refetchInterval: 5000 });
  const ontology = useQuery({ queryKey: ["ontology"], queryFn: api.ontology });
  const rs = useQuery({
    queryKey: ["realsense"],
    queryFn: api.realsenseStatus,
    refetchInterval: 2000
  });

  const [importPath, setImportPath] = useState("");
  const [recordSeconds, setRecordSeconds] = useState(10);
  const [recProgress, setRecProgress] = useState<string | null>(null);
  const [recPreview, setRecPreview] = useState<string | null>(null);
  const [recActiveClip, setRecActiveClip] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  function clearPreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setRecPreview(null);
  }

  function teardownWs() {
    if (wsRef.current) {
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      try {
        wsRef.current.close();
      } catch {
        /* ignore */
      }
      wsRef.current = null;
    }
  }

  useEffect(() => {
    return () => {
      teardownWs();
      clearPreview();
    };
  }, []);

  const importMut = useMutation({
    mutationFn: (p: string) =>
      p.endsWith(".bag") ? api.importBag(p) : api.importFolder(p),
    onSuccess: () => {
      setImportPath("");
      qc.invalidateQueries({ queryKey: ["clips"] });
    }
  });

  function attachRecorderWs(clipId: string) {
    teardownWs();
    setRecActiveClip(clipId);
    const ws = captureWebSocket(clipId);
    ws.binaryType = "blob";
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data);
        if (m.event === "frame") setRecProgress(`recording: ${m.written} frames`);
        else if (m.event === "started") setRecProgress(`recording ${clipId}…`);
        else if (m.event === "snapshot") setRecProgress(`recording: ${m.written} frames`);
        else if (m.event === "completed") {
          setRecProgress(`done: ${m.frame_count} frames`);
          qc.invalidateQueries({ queryKey: ["clips"] });
          qc.invalidateQueries({ queryKey: ["realsense"] });
          setRecActiveClip(null);
          // keep last preview frame visible briefly, then clear
          setTimeout(clearPreview, 1500);
          teardownWs();
        } else if (m.event === "error") {
          setRecProgress(`error: ${m.detail}`);
          setRecActiveClip(null);
          clearPreview();
          teardownWs();
        }
      } else {
        const blob = ev.data as Blob;
        const url = URL.createObjectURL(blob);
        const prev = previewUrlRef.current;
        previewUrlRef.current = url;
        setRecPreview(url);
        if (prev) URL.revokeObjectURL(prev);
      }
    };
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
    };
  }

  const recordMut = useMutation({
    mutationFn: (s: number) => api.record(s),
    onSuccess: (clip) => {
      setRecProgress(`recording ${clip.clip_id}…`);
      attachRecorderWs(clip.clip_id);
    }
  });

  const stopMut = useMutation({ mutationFn: api.stopRecord });
  const deleteMut = useMutation({
    mutationFn: api.deleteClip,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["clips"] })
  });

  // If the page reloaded mid-recording, attach to the running session.
  useEffect(() => {
    const ongoing = rs.data?.recording_clip_id;
    if (ongoing && !wsRef.current && !recActiveClip) {
      setRecProgress(`recording ${ongoing}…`);
      attachRecorderWs(ongoing);
    }
  }, [rs.data?.recording_clip_id, recActiveClip]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-slate-400">Clips</h2>
          <div className="text-xs text-slate-500">
            {clips.data?.length ?? 0} total ·{" "}
            {rs.data?.available ? "RealSense detected" : "no RealSense"} ·{" "}
            {rs.data?.busy ? "recording" : "idle"}
          </div>
        </div>
        {clips.isLoading ? (
          <div className="text-slate-500">Loading…</div>
        ) : clips.error ? (
          <div className="text-rose-400">{(clips.error as Error).message}</div>
        ) : !clips.data?.length ? (
          <div className="rounded-lg border border-dashed border-slate-700 p-12 text-center text-slate-500">
            No clips yet. Use the panel on the right to import a folder, a .bag, or record live.
          </div>
        ) : (
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {clips.data.map((c) => (
              <li
                key={c.clip_id}
                className="rounded-lg border border-slate-800 bg-slate-900 p-3 hover:border-slate-600 flex flex-col gap-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <Link
                    to={`/label/${c.clip_id}`}
                    className="font-mono text-sm hover:text-emerald-300"
                  >
                    {c.clip_id}
                  </Link>
                  <button
                    onClick={() => deleteMut.mutate(c.clip_id)}
                    className="text-xs text-slate-500 hover:text-rose-400"
                  >
                    delete
                  </button>
                </div>
                <div className="text-xs text-slate-400">
                  {c.frame_count} fr · {c.width}×{c.height} · {c.fps.toFixed(1)} fps · {c.source}
                </div>
                {c.frame_count > 0 ? (
                  <img
                    src={`/api/clips/${c.clip_id}/frames/0`}
                    alt={c.clip_id}
                    className="aspect-video object-cover rounded border border-slate-800"
                  />
                ) : (
                  <div className="aspect-video rounded border border-dashed border-slate-800 grid place-items-center text-xs text-slate-600">
                    no frames yet
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <aside className="space-y-6">
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-3">
          <h3 className="text-sm uppercase tracking-wider text-slate-400 mb-2">Capture</h3>
          <div className="space-y-2">
            <label className="block text-xs text-slate-400">Import folder or .bag path</label>
            <input
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              placeholder="/abs/path/to/frames or /path/to/clip.bag"
              className="w-full rounded bg-slate-950 border border-slate-700 px-2 py-1 text-sm"
            />
            <button
              disabled={!importPath || importMut.isPending}
              onClick={() => importMut.mutate(importPath)}
              className="w-full rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-50 px-3 py-1.5 text-sm"
            >
              {importMut.isPending ? "Importing…" : "Import"}
            </button>
            {importMut.error ? (
              <div className="text-xs text-rose-400">{(importMut.error as Error).message}</div>
            ) : null}
          </div>
          <div className="mt-4 border-t border-slate-800 pt-3 space-y-2">
            <label className="block text-xs text-slate-400">Record live (max-seconds; Stop ends early)</label>
            <div className="flex gap-2">
              <input
                type="number"
                min={1}
                max={60}
                value={recordSeconds}
                onChange={(e) => setRecordSeconds(Number(e.target.value))}
                className="w-20 rounded bg-slate-950 border border-slate-700 px-2 py-1 text-sm"
                disabled={rs.data?.busy ?? false}
              />
              <button
                disabled={!rs.data?.available || rs.data?.busy || recordMut.isPending}
                onClick={() => recordMut.mutate(recordSeconds)}
                className="flex-1 rounded bg-rose-700 hover:bg-rose-600 disabled:opacity-40 px-3 py-1.5 text-sm"
              >
                {rs.data?.busy ? "Recording…" : `Record ${recordSeconds}s`}
              </button>
              <button
                disabled={!rs.data?.busy && !recActiveClip}
                onClick={() => stopMut.mutate()}
                className="rounded bg-amber-700 hover:bg-amber-600 disabled:opacity-30 px-3 py-1.5 text-sm"
              >
                Stop
              </button>
            </div>
            {recProgress ? <div className="text-xs text-slate-400">{recProgress}</div> : null}
          </div>

          {rs.data?.available ? (
            <div className="mt-3 relative">
              <img
                src="/api/realsense/stream.mjpg"
                className={`w-full rounded border ${rs.data?.busy ? "border-rose-700" : "border-slate-800"}`}
                alt="RealSense preview"
                onError={(e) => {
                  (e.target as HTMLImageElement).alt =
                    "Preview failed. Refresh the page; if it persists, check the server log.";
                }}
              />
              {rs.data?.busy ? (
                <span className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider bg-rose-700 text-white">
                  ● REC
                </span>
              ) : null}
              <div className="text-xs text-slate-500 mt-1">
                live preview · MJPEG {rs.data?.subscribers ? `· ${rs.data.subscribers} client(s)` : ""}
              </div>
              {rs.data?.busy && recPreview ? (
                <details className="mt-2 text-xs text-slate-400">
                  <summary className="cursor-pointer hover:text-slate-200">
                    show last saved frame
                  </summary>
                  <img
                    src={recPreview}
                    className="w-full rounded border border-slate-800 mt-2"
                    alt="last saved frame"
                  />
                </details>
              ) : null}
            </div>
          ) : (
            <div className="mt-3 text-xs text-amber-400">
              No RealSense detected. Plug it in or check pyrealsense2 is installed in this venv.
            </div>
          )}
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-3">
          <h3 className="text-sm uppercase tracking-wider text-slate-400 mb-2">Ontology</h3>
          {ontology.isLoading ? (
            <div className="text-slate-500">Loading…</div>
          ) : ontology.error ? (
            <div className="text-rose-400">{(ontology.error as Error).message}</div>
          ) : (
            <div>
              <div className="text-xs text-slate-500 mb-2">
                {ontology.data?.labels.length} classes · hash {ontology.data?.hash}
              </div>
              <ul className="space-y-1 text-sm">
                {ontology.data?.labels.map((label, i) => (
                  <li key={label} className="flex justify-between gap-2">
                    <span className="text-slate-200">{label}</span>
                    <span className="text-slate-500 truncate" title={ontology.data?.prompts[i]}>
                      {ontology.data?.prompts[i]}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </aside>
    </div>
  );
}
