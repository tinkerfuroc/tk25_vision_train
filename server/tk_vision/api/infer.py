from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.infer_service import InferJob, InferStatus, load_predictions

log = logging.getLogger("tk_vision.api.infer")

router = APIRouter(prefix="/api", tags=["infer"])


class InferRequest(BaseModel):
    weights_path: str
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.5, ge=0.0, le=1.0)


class InferStatusResponse(BaseModel):
    job_id: str
    run_id: str
    clip_id: str
    status: InferStatus
    error: str | None
    done_frames: int
    total_frames: int
    predictions_path: str | None


def _manager(request: Request):
    mgr = getattr(request.app.state, "infer", None)
    if mgr is None:
        raise HTTPException(503, "Inference manager not initialized")
    return mgr


def _to_status(job: InferJob) -> InferStatusResponse:
    return InferStatusResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        clip_id=job.clip_id,
        status=job.status,
        error=job.error,
        done_frames=job.done_frames,
        total_frames=job.total_frames,
        predictions_path=job.predictions_path,
    )


@router.post("/runs/{run_id}/infer/{clip_id}", response_model=InferStatusResponse)
async def start_infer(
    run_id: str, clip_id: str, payload: InferRequest, request: Request
) -> InferStatusResponse:
    mgr = _manager(request)
    weights = Path(payload.weights_path)
    settings = request.app.state.settings
    if not weights.is_absolute():
        weights = settings.resolve(payload.weights_path)
    try:
        job = mgr.start(
            run_id=run_id,
            clip_id=clip_id,
            weights_path=weights,
            conf=payload.conf,
            iou=payload.iou,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return _to_status(job)


@router.get("/runs/{run_id}/infer/{clip_id}/{job_id}", response_model=InferStatusResponse)
async def get_infer(
    run_id: str, clip_id: str, job_id: str, request: Request
) -> InferStatusResponse:
    mgr = _manager(request)
    job = mgr.get(job_id)
    if job is None or job.run_id != run_id or job.clip_id != clip_id:
        raise HTTPException(404, "job not found")
    return _to_status(job)


@router.delete("/runs/{run_id}/infer/{clip_id}/{job_id}", response_model=InferStatusResponse)
async def cancel_infer(
    run_id: str, clip_id: str, job_id: str, request: Request
) -> InferStatusResponse:
    mgr = _manager(request)
    job = mgr.get(job_id)
    if job is None or job.run_id != run_id or job.clip_id != clip_id:
        raise HTTPException(404, "job not found")
    mgr.cancel(job_id)
    return _to_status(job)


class InferPredictionsResponse(BaseModel):
    run_id: str
    clip_id: str
    weights_path: str
    conf: float
    iou: float
    frame_count: int
    deleted_frames: list[int]
    predictions: dict[str, list[dict]]


@router.get("/runs/{run_id}/predictions/{clip_id}", response_model=InferPredictionsResponse)
async def get_predictions(
    run_id: str,
    clip_id: str,
    request: Request,
    frame: int | None = None,
    from_idx: int | None = None,
    to_idx: int | None = None,
) -> InferPredictionsResponse:
    """Return the full predictions manifest, or a sliced view.

    `?frame=N` returns only frame N's detections. `?from_idx=A&to_idx=B`
    returns frames in [A, B). Out-of-range frames are silently empty.
    """
    store = request.app.state.store
    data = load_predictions(store, run_id, clip_id)
    if data is None:
        raise HTTPException(404, "predictions not found; run inference first")
    if frame is not None:
        preds = data.get("predictions", {})
        data = {**data, "predictions": {str(frame): preds.get(str(frame), [])}}
    elif from_idx is not None or to_idx is not None:
        lo = from_idx if from_idx is not None else 0
        hi = to_idx if to_idx is not None else data["frame_count"]
        preds = data.get("predictions", {})
        data = {
            **data,
            "predictions": {k: v for k, v in preds.items() if lo <= int(k) < hi},
        }
    return InferPredictionsResponse(**data)
