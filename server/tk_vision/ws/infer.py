from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ._drain import drain_job_ws, resolve_job_or_close

router = APIRouter(prefix="/ws/infer", tags=["ws"])


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
