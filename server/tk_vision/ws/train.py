from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ._drain import drain_job_ws, resolve_job_or_close

router = APIRouter(prefix="/ws/train", tags=["ws"])


@router.websocket("/{run_id}/{job_id}")
async def train_ws(websocket: WebSocket, run_id: str, job_id: str) -> None:
    """Stream TrainJob log + lifecycle events.

    Server → client (one JSON per event):
        {"event": "log", "line": str}
        {"event": "started", "cmd": str}
        {"event": "done"|"cancelled"|"error", ...}
    """
    await websocket.accept()
    job = await resolve_job_or_close(websocket, "train", job_id, run_id=run_id)
    if job is None:
        return
    await drain_job_ws(
        websocket,
        job,
        ready_payload={
            "event": "ready",
            "job_id": job.job_id,
            "status": job.status,
            "log_tail": list(job.log_lines)[-20:],
        },
        job_label="train",
    )
