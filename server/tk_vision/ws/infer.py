from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ._drain import drain_job_ws, resolve_job_or_close

log = logging.getLogger("tk_vision.ws.infer")

router = APIRouter(prefix="/ws/infer", tags=["ws"])


@router.websocket("/live/{job_id}")
async def live_infer_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream live inference results (detections per frame)."""
    await websocket.accept()
    mgr = getattr(websocket.app.state, "infer", None)
    if mgr is None:
        await websocket.send_json({"event": "error", "detail": "infer manager not available"})
        await websocket.close()
        return

    job = mgr.get_live(job_id)
    if job is None:
        log.warning("live_infer_ws: job %s not found", job_id)
        await websocket.send_json({"event": "error", "detail": "job not found"})
        await websocket.close()
        return

    log.info("live_infer_ws: connected to job %s (status=%s)", job_id, job.status)
    await websocket.send_json({
        "event": "ready",
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
    })

    try:
        while True:
            try:
                evt = await asyncio.wait_for(job.progress.get(), timeout=30.0)
            except asyncio.TimeoutError:
                if job.status in ("done", "error", "cancelled"):
                    break
                await websocket.send_json({"event": "ping", "status": job.status})
                continue

            await websocket.send_json(evt)

            if evt.get("event") in ("done", "error", "cancelled"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("live_infer_ws error: %s", e)
        await websocket.send_json({"event": "error", "detail": str(e)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/{run_id}/{clip_id}/{job_id}")
async def infer_ws(
    websocket: WebSocket, run_id: str, clip_id: str, job_id: str
) -> None:
    """Stream InferJob progress events."""
    await websocket.accept()
    job = await resolve_job_or_close(
        websocket, "infer", job_id, run_id=run_id, clip_id=clip_id
    )
    if job is None:
        return
    await drain_job_ws(
        websocket,
        job,
        ready_payload={
            "event": "ready",
            "job_id": job.job_id,
            "status": job.status,
            "done": job.done_frames,
            "total": job.total_frames,
        },
        job_label="infer",
    )


@router.websocket("/{clip_id}/{job_id}")
async def infer_ws_no_run(
    websocket: WebSocket, clip_id: str, job_id: str
) -> None:
    """Stream InferJob progress events (temp_infer run)."""
    log.info("infer_ws_no_run: connecting clip_id=%s, job_id=%s", clip_id, job_id)
    await websocket.accept()
    job = await resolve_job_or_close(
        websocket, "infer", job_id, run_id="temp_infer", clip_id=clip_id
    )
    if job is None:
        log.warning("infer_ws_no_run: job %s not found for clip %s", job_id, clip_id)
        return
    log.info("infer_ws_no_run: job %s found, draining", job_id)
    await drain_job_ws(
        websocket,
        job,
        ready_payload={
            "event": "ready",
            "job_id": job.job_id,
            "status": job.status,
            "done": job.done_frames,
            "total": job.total_frames,
        },
        job_label="infer",
    )
