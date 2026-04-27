"""Shared WebSocket drain loop for background-job queues.

All long-running jobs (train, infer, propagate) emit progress dicts onto an
`asyncio.Queue` and flip `status` between `pending → running → done|error|
cancelled`. The WS contract is the same for every endpoint:

    1. accept the socket
    2. send a `ready` payload (caller-supplied)
    3. drain the queue until a terminal event or status is observed
    4. close the socket

`drain_job_ws` factors steps 3+ out so each ws router becomes a 5-line
wrapper that just constructs the `ready` payload.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..services.train_service import TERMINAL_STATUSES

log = logging.getLogger("tk_vision.ws.drain")


async def resolve_job_or_close(
    websocket: WebSocket, mgr_attr: str, job_id: str, **must_match: Any
) -> Any:
    """Look up a job on `app.state.<mgr_attr>` by `job_id`; if missing or any
    `must_match` field doesn't match (e.g. `run_id="r1"`), send an error
    payload and close. Returns the job or `None`."""
    mgr = getattr(websocket.app.state, mgr_attr, None)
    job = mgr.get(job_id) if mgr else None
    if job is None or any(getattr(job, k, None) != v for k, v in must_match.items()):
        await websocket.send_json({"event": "error", "detail": "job not found"})
        await websocket.close()
        return None
    return job


async def drain_job_ws(
    websocket: WebSocket,
    job: Any,
    *,
    ready_payload: dict,
    job_label: str = "job",
    timeout_s: float = 30.0,
) -> None:
    """Send `ready_payload`, then forward queue events until terminal.

    `job` must expose:
      - `status: str`
      - `progress: asyncio.Queue[dict]`
    """
    await websocket.send_json(ready_payload)
    try:
        while True:
            if job.status in TERMINAL_STATUSES and job.progress.empty():
                await websocket.send_json({"event": job.status, "final": True})
                break
            try:
                evt = await asyncio.wait_for(job.progress.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "ping", "status": job.status})
                continue
            await websocket.send_json(evt)
            if evt.get("event") in TERMINAL_STATUSES:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("%s ws crashed: %s", job_label, e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
