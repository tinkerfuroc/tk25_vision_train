from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..data.manifest import ClipMeta

router = APIRouter(prefix="/api/clips", tags=["clips"])
log = logging.getLogger("tk_vision.api.clips")


class ClipSummary(BaseModel):
    clip_id: str
    source: str
    frame_count: int
    fps: float
    width: int
    height: int
    bag_path: str | None = None


class RecordRequest(BaseModel):
    max_seconds: float = 30.0


class ImportRequest(BaseModel):
    folder_path: str | None = None
    bag_path: str | None = None


def _capture(request: Request):
    cap = getattr(request.app.state, "capture", None)
    if cap is None:
        raise HTTPException(503, "Capture service not initialized")
    return cap


@router.get("", response_model=list[ClipSummary])
async def list_clips(request: Request) -> list[ClipSummary]:
    store = request.app.state.store
    out: list[ClipSummary] = []
    for cid in store.list_clip_ids():
        try:
            meta = store.read_meta(cid)
        except FileNotFoundError:
            log.warning("Skipping clip without metadata: %s", cid)
            continue
        except ValueError as e:
            # Corrupt manifests / invalid clip metadata should not break
            # listing of all clips.
            log.warning("Skipping malformed clip %s: %s", cid, e)
            continue
        out.append(ClipSummary(**meta.model_dump()))
    return out


@router.get("/{clip_id}", response_model=ClipSummary)
async def get_clip(clip_id: str, request: Request) -> ClipSummary:
    store = request.app.state.store
    if not store.clip_dir(clip_id).exists():
        raise HTTPException(404, "Clip not found")
    return ClipSummary(**store.read_meta(clip_id).model_dump())


@router.delete("/{clip_id}")
async def delete_clip(clip_id: str, request: Request) -> dict:
    store = request.app.state.store
    if not store.clip_dir(clip_id).exists():
        raise HTTPException(404, "Clip not found")
    store.delete_clip(clip_id)
    return {"deleted": clip_id}


@router.get("/{clip_id}/frames/{idx}", response_class=FileResponse)
async def get_frame(clip_id: str, idx: int, request: Request) -> FileResponse:
    store = request.app.state.store
    p = store.frames_dir(clip_id) / f"{idx:06d}.jpg"
    if not p.is_file():
        raise HTTPException(404, f"Frame {idx} not found in {clip_id}")
    return FileResponse(str(p), media_type="image/jpeg")


@router.post("/record", response_model=ClipSummary)
async def record(payload: RecordRequest, request: Request) -> ClipSummary:
    cap = _capture(request)
    if cap.is_recording():
        raise HTTPException(409, "A recording is already in progress")
    settings = request.app.state.settings
    seconds = min(max(0.1, payload.max_seconds), float(settings.capture.max_clip_seconds))
    try:
        clip_id = await cap.start_record(max_seconds=seconds)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    return ClipSummary(
        clip_id=clip_id,
        source="live",
        frame_count=0,
        fps=float(settings.capture.fps),
        width=settings.capture.resolution[0],
        height=settings.capture.resolution[1],
        bag_path=None,
    )


@router.post("/record/stop")
async def stop_record(request: Request) -> dict:
    cap = _capture(request)
    await cap.stop_record()
    return {"stopped": True}


@router.post("/import", response_model=ClipSummary)
async def import_clip(payload: ImportRequest, request: Request) -> ClipSummary:
    cap = _capture(request)
    if payload.folder_path:
        meta = cap.import_folder(payload.folder_path)
    elif payload.bag_path:
        meta = cap.import_bag(payload.bag_path)
    else:
        raise HTTPException(400, "Provide folder_path or bag_path")
    return ClipSummary(**meta.model_dump())
