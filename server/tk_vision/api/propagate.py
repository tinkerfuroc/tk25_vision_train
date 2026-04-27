from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..api.label import _clip_detail, _label_service, _require_clip
from .label import ClipDetail

log = logging.getLogger("tk_vision.propagate")

router = APIRouter(prefix="/api/clips", tags=["propagate"])


@dataclass
class PropagateJob:
    job_id: str
    clip_id: str
    start: int
    end: int
    chunk_size: int
    chunk_overlap: int
    respect_edits: bool
    status: str = "pending"  # pending | running | done | error | cancelled
    error: Optional[str] = None
    done_frames: int = 0
    total_frames: int = 0
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    progress: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    task: Optional[asyncio.Task] = None


class PropagateRequest(BaseModel):
    start: int = 0
    end: Optional[int] = None
    chunk_size: int = 50
    chunk_overlap: int = 4
    respect_edits: bool = True


class PropagateResponse(BaseModel):
    job_id: str
    clip_id: str
    start: int
    end: int
    total_frames: int


def _jobs(request: Request) -> dict[str, PropagateJob]:
    if not hasattr(request.app.state, "propagate_jobs"):
        request.app.state.propagate_jobs = {}
    return request.app.state.propagate_jobs


@router.post("/{clip_id}/propagate", response_model=PropagateResponse)
async def start_propagate(
    clip_id: str, payload: PropagateRequest, request: Request
) -> PropagateResponse:
    svc = _label_service(request)
    store = _require_clip(request, clip_id)
    clip = store.read_clip(clip_id)
    end = payload.end if payload.end is not None else clip.frame_count
    if payload.start < 0 or end > clip.frame_count or end <= payload.start:
        raise HTTPException(400, f"invalid range [{payload.start}, {end}) for {clip.frame_count} frames")
    if not clip.tracks:
        raise HTTPException(400, "clip has no tracks; seed first")

    jobs = _jobs(request)
    job = PropagateJob(
        job_id=uuid.uuid4().hex[:8],
        clip_id=clip_id,
        start=payload.start,
        end=end,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        respect_edits=payload.respect_edits,
        total_frames=end - payload.start,
    )
    jobs[job.job_id] = job

    async def progress_cb(evt: dict) -> None:
        if evt.get("event") == "frame" and "done" in evt:
            job.done_frames = evt["done"]
        try:
            job.progress.put_nowait(evt)
        except asyncio.QueueFull:
            # Drop oldest then push.
            try:
                job.progress.get_nowait()
            except Exception:
                pass
            try:
                job.progress.put_nowait(evt)
            except Exception:
                pass

    async def runner() -> None:
        job.status = "running"
        try:
            await svc.propagate_clip(
                clip_id,
                start=payload.start,
                end=end,
                chunk_size=payload.chunk_size,
                chunk_overlap=payload.chunk_overlap,
                respect_edits=payload.respect_edits,
                progress_cb=progress_cb,
                cancel_cb=lambda: job.cancel.is_set(),
            )
            job.status = "cancelled" if job.cancel.is_set() else "done"
        except Exception as e:  # noqa: BLE001
            log.exception("propagate job %s failed", job.job_id)
            job.status = "error"
            job.error = str(e)
        finally:
            try:
                job.progress.put_nowait({"event": "end", "status": job.status, "error": job.error})
            except Exception:
                pass

    job.task = asyncio.create_task(runner())
    return PropagateResponse(
        job_id=job.job_id,
        clip_id=clip_id,
        start=payload.start,
        end=end,
        total_frames=job.total_frames,
    )


class PropagateStatus(BaseModel):
    job_id: str
    clip_id: str
    status: str
    done_frames: int
    total_frames: int
    error: Optional[str]
    detail: Optional[ClipDetail]


@router.get("/{clip_id}/propagate/{job_id}", response_model=PropagateStatus)
async def get_propagate(
    clip_id: str, job_id: str, request: Request
) -> PropagateStatus:
    job = _jobs(request).get(job_id)
    if job is None or job.clip_id != clip_id:
        raise HTTPException(404, "job not found")
    store = request.app.state.store
    clip = store.read_clip(clip_id)
    return PropagateStatus(
        job_id=job.job_id,
        clip_id=clip_id,
        status=job.status,
        done_frames=job.done_frames,
        total_frames=job.total_frames,
        error=job.error,
        detail=_clip_detail(clip),
    )


@router.delete("/{clip_id}/propagate/{job_id}", response_model=PropagateStatus)
async def cancel_propagate(
    clip_id: str, job_id: str, request: Request
) -> PropagateStatus:
    job = _jobs(request).get(job_id)
    if job is None or job.clip_id != clip_id:
        raise HTTPException(404, "job not found")
    if job.status == "running":
        job.cancel.set()
    return await get_propagate(clip_id, job_id, request)
