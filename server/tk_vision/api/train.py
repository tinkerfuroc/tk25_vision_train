from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.train_service import TERMINAL_STATUSES, TrainJob, TrainStatus as TrainStatusLiteral

log = logging.getLogger("tk_vision.api.train")

router = APIRouter(prefix="/api/runs", tags=["train"])


class TrainRequest(BaseModel):
    base_weights: str | None = None
    epochs: int | None = Field(None, gt=0)
    imgsz: int | None = Field(None, gt=0)
    batch: int | None = Field(None, gt=0)
    patience: int | None = Field(None, ge=0)
    device: str | None = None


class TrainStatus(BaseModel):
    job_id: str
    run_id: str
    status: TrainStatusLiteral
    pid: int | None
    return_code: int | None
    error: str | None
    log_tail: list[str]
    metrics: dict | None


def _manager(request: Request):
    mgr = getattr(request.app.state, "train", None)
    if mgr is None:
        raise HTTPException(503, "Train manager not initialized")
    return mgr


_NO_METRICS: dict = {}


def _metrics(job: TrainJob) -> dict | None:
    if job.metrics_cache is not None:
        return job.metrics_cache or None
    p = job.project_dir / job.name / "metrics.json"
    if not p.exists():
        if job.status in TERMINAL_STATUSES:
            job.metrics_cache = _NO_METRICS
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("failed to read metrics.json for %s: %s", job.job_id, e)
        return None
    if job.status in TERMINAL_STATUSES:
        job.metrics_cache = data
    return data


def _to_status(job: TrainJob) -> TrainStatus:
    return TrainStatus(
        job_id=job.job_id,
        run_id=job.run_id,
        status=job.status,
        pid=job.pid,
        return_code=job.return_code,
        error=job.error,
        log_tail=list(job.log_lines)[-50:],
        metrics=_metrics(job),
    )


@router.post("/{run_id}/train", response_model=TrainStatus)
async def start_train(run_id: str, payload: TrainRequest, request: Request) -> TrainStatus:
    settings = request.app.state.settings
    store = request.app.state.store
    mgr = _manager(request)
    run_dir = store.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {run_id}")
    overrides = payload.model_dump(exclude_none=True, exclude={"device"})
    cfg = settings.train.model_copy(update=overrides)
    try:
        job = mgr.start(run_id=run_id, run_dir=run_dir, cfg=cfg, device=payload.device or "")
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    return _to_status(job)


@router.get("/{run_id}/train", response_model=list[TrainStatus])
async def list_train_jobs(run_id: str, request: Request) -> list[TrainStatus]:
    mgr = _manager(request)
    return [_to_status(j) for j in mgr.list_jobs(run_id)]


@router.get("/{run_id}/train/{job_id}", response_model=TrainStatus)
async def get_train(run_id: str, job_id: str, request: Request) -> TrainStatus:
    mgr = _manager(request)
    job = mgr.get(job_id)
    if job is None or job.run_id != run_id:
        raise HTTPException(404, "job not found")
    return _to_status(job)


@router.delete("/{run_id}/train/{job_id}", response_model=TrainStatus)
async def cancel_train(run_id: str, job_id: str, request: Request) -> TrainStatus:
    mgr = _manager(request)
    job = mgr.get(job_id)
    if job is None or job.run_id != run_id:
        raise HTTPException(404, "job not found")
    mgr.cancel(job_id)
    return _to_status(job)
