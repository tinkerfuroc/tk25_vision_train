from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger("tk_vision.ws.propagate")
router = APIRouter(prefix="/ws/propagate", tags=["ws"])


@router.websocket("/{clip_id}/{job_id}")
async def propagate_ws(websocket: WebSocket, clip_id: str, job_id: str) -> None:
    """Stream PropagateJob progress events to the client.

    Server → client (one JSON per event):
        {"event": "frame", "frame_idx": int, "done": int, "total": int, "source": "propagated"|"propagated_overlap"}
        {"event": "chunk_done", "chunk_end": int}
        {"event": "frame", "skipped": "edited"|"missing_jpeg", ...}
        {"event": "end", "status": "done"|"cancelled"|"error", "error": str|null}
    """
    await websocket.accept()
    jobs = getattr(websocket.app.state, "propagate_jobs", None)
    job = jobs.get(job_id) if jobs else None
    if job is None or job.clip_id != clip_id:
        await websocket.send_json({"event": "error", "detail": "job not found"})
        await websocket.close()
        return

    await websocket.send_json(
        {
            "event": "ready",
            "job_id": job.job_id,
            "status": job.status,
            "done_frames": job.done_frames,
            "total_frames": job.total_frames,
        }
    )
    try:
        while True:
            try:
                evt = await asyncio.wait_for(job.progress.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat to keep the socket alive.
                await websocket.send_json({"event": "ping", "status": job.status})
                if job.status in ("done", "cancelled", "error"):
                    break
                continue
            await websocket.send_json(evt)
            if evt.get("event") == "end":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("propagate ws crashed: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
