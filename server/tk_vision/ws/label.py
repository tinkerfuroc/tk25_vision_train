from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger("tk_vision.ws.label")
router = APIRouter(prefix="/ws/label", tags=["ws"])


@router.websocket("/{clip_id}")
async def label_ws(websocket: WebSocket, clip_id: str) -> None:
    """Bidirectional label channel.

    Client → server: refine requests as JSON.
        {"op": "refine", "frame_idx": <int>,
         "track_id": <int|null>, "class_id": <int|null>, "label": <str|null>,
         "points": [[x,y,label],...]?, "box": [x1,y1,x2,y2]? }

    Server → client:
        {"event": "ready", "clip": {... clip detail ...}}
        {"event": "refined", "track_id": ..., "frame_idx": ...}
        {"event": "error", "detail": ...}
    """
    await websocket.accept()
    svc = getattr(websocket.app.state, "label", None)
    store = getattr(websocket.app.state, "store", None)
    if svc is None or store is None:
        await websocket.send_json({"event": "error", "detail": "Label service not ready"})
        await websocket.close()
        return
    if not store.clip_dir(clip_id).exists():
        await websocket.send_json({"event": "error", "detail": "Clip not found"})
        await websocket.close()
        return

    from ..api.label import _clip_detail

    try:
        clip = store.read_clip(clip_id)
        await websocket.send_json({"event": "ready", "clip": _clip_detail(clip).model_dump()})
    except Exception as e:  # noqa: BLE001
        await websocket.send_json({"event": "error", "detail": f"load failed: {e}"})
        await websocket.close()
        return

    try:
        while True:
            msg = await websocket.receive_json()
            op = msg.get("op")
            if op == "refine":
                try:
                    clip, track, mask = await svc.refine_frame(
                        clip_id,
                        int(msg["frame_idx"]),
                        track_id=msg.get("track_id"),
                        points=msg.get("points"),
                        box=tuple(msg["box"]) if msg.get("box") else None,
                        class_id=msg.get("class_id"),
                        label=msg.get("label"),
                    )
                    await websocket.send_json(
                        {
                            "event": "refined",
                            "track_id": track.track_id,
                            "frame_idx": mask.frame_idx,
                            "label": track.label,
                            "class_id": track.class_id,
                            "clip": _clip_detail(clip).model_dump(),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    await websocket.send_json({"event": "error", "detail": str(e)})
            elif op == "seed":
                try:
                    clip = await svc.seed_first_frame(clip_id)
                    await websocket.send_json(
                        {
                            "event": "seeded",
                            "clip": _clip_detail(clip).model_dump(),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    await websocket.send_json({"event": "error", "detail": str(e)})
            elif op == "ping":
                await websocket.send_json({"event": "pong"})
            else:
                await websocket.send_json({"event": "error", "detail": f"unknown op: {op}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("label ws crashed: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
