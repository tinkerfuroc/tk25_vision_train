from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws/capture", tags=["ws"])

QueueItem = tuple[Literal["json", "bytes"], Any]


@router.websocket("/{clip_id}")
async def capture_progress(websocket: WebSocket, clip_id: str) -> None:
    await websocket.accept()
    cap = getattr(websocket.app.state, "capture", None)
    if cap is None:
        await websocket.send_json({"event": "error", "detail": "Capture service not ready"})
        await websocket.close()
        return

    queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=128)

    async def progress_cb(payload: dict) -> None:
        try:
            queue.put_nowait(("json", payload))
        except asyncio.QueueFull:
            pass

    async def frame_cb(jpeg: bytes) -> None:
        # Drop the oldest preview frame if we're behind, then push the latest.
        try:
            queue.put_nowait(("bytes", jpeg))
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(("bytes", jpeg))
            except asyncio.QueueFull:
                pass

    await cap.subscribe_progress(clip_id, progress_cb)
    await cap.subscribe_frames(clip_id, frame_cb)
    try:
        rec = cap.recorder_state()
        if rec and rec.clip_id == clip_id:
            await websocket.send_json(
                {"event": "snapshot", "clip_id": clip_id, "written": rec.written, "stopped": rec.stopped}
            )
        while True:
            kind, payload = await queue.get()
            if kind == "bytes":
                await websocket.send_bytes(payload)
            else:
                await websocket.send_json(payload)
                if payload.get("event") in ("completed", "error"):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await cap.unsubscribe_progress(clip_id, progress_cb)
        await cap.unsubscribe_frames(clip_id, frame_cb)
        try:
            await websocket.close()
        except Exception:
            pass
