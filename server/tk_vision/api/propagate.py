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
    chunk_size: int = 25  # Reduced from 50 to avoid CUDA OOM
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

    # Enforce max chunk_size to prevent CUDA OOM
    # 50 frames at 1008x1008 bfloat16 can exhaust 8-12GB VRAM
    MAX_CHUNK_SIZE = 20
    chunk_size = min(payload.chunk_size, MAX_CHUNK_SIZE)
    if chunk_size != payload.chunk_size:
        log.warning(
            "propagate: chunk_size %d exceeds max %d, reducing",
            payload.chunk_size, MAX_CHUNK_SIZE
        )

    jobs = _jobs(request)

    # Check for existing running job for this clip - prevent concurrent propagates
    for existing_job in jobs.values():
        if existing_job.clip_id == clip_id and existing_job.status in ("pending", "running"):
            raise HTTPException(
                409,
                f"Propagate job {existing_job.job_id} already running for clip {clip_id}. "
                f"Wait for it to complete or cancel it first."
            )

    job = PropagateJob(
        job_id=uuid.uuid4().hex[:8],
        clip_id=clip_id,
        start=payload.start,
        end=end,
        chunk_size=chunk_size,
        chunk_overlap=payload.chunk_overlap,
        respect_edits=payload.respect_edits,
        total_frames=end - payload.start,
    )
    jobs[job.job_id] = job
    log.info("Created propagate job %s for clip %s (%d frames)", job.job_id, clip_id, job.total_frames)

    async def progress_cb(evt: dict) -> None:
        if evt.get("event") == "frame" and "done" in evt:
            job.done_frames = evt["done"]
            log.debug("propagate %s: frame %d/%d", job.job_id, job.done_frames, job.total_frames)
        elif evt.get("event") == "chunk_done":
            log.info("propagate %s: chunk done at frame %d", job.job_id, evt.get("chunk_end"))
        elif evt.get("event") == "chunk_skipped":
            log.warning("propagate %s: chunk skipped at %d: %s", job.job_id, evt.get("chunk_start"), evt.get("reason"))
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
        log.info("propagate job %s started (chunk_size=%d)", job.job_id, chunk_size)
        try:
            await svc.propagate_clip(
                clip_id,
                start=payload.start,
                end=end,
                chunk_size=chunk_size,
                chunk_overlap=payload.chunk_overlap,
                respect_edits=payload.respect_edits,
                progress_cb=progress_cb,
                cancel_cb=lambda: job.cancel.is_set(),
            )
            job.status = "cancelled" if job.cancel.is_set() else "done"
            log.info("propagate job %s completed: %s", job.job_id, job.status)
        except Exception as e:  # noqa: BLE001
            log.exception("propagate job %s failed: %s", job.job_id, e)
            job.status = "error"
            job.error = str(e)
        finally:
            try:
                job.progress.put_nowait({"event": "end", "status": job.status, "error": job.error})
            except Exception:
                pass

    def _log_task_exception(task: asyncio.Task) -> None:
        """Log any exception from the background propagate task."""
        try:
            exc = task.exception()
            if exc:
                log.exception("propagate task %s crashed: %s", job.job_id, exc)
        except asyncio.CancelledError:
            log.info("propagate task %s was cancelled", job.job_id)

    job.task = asyncio.create_task(runner())
    job.task.add_done_callback(_log_task_exception)
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
