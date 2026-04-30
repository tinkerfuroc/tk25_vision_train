from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.infer_service import InferJob, InferStatus, LiveInferJob, load_predictions

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


class LiveInferStatusResponse(BaseModel):
    job_id: str
    status: InferStatus
    error: str | None
    frame_count: int


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


def _to_live_status(job: LiveInferJob) -> LiveInferStatusResponse:
    return LiveInferStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        frame_count=job.frame_count,
    )


@router.post("/live-infer", response_model=LiveInferStatusResponse)
async def start_live_infer(payload: InferRequest, request: Request) -> LiveInferStatusResponse:
    """Start real-time inference on live camera feed."""
    mgr = _manager(request)
    weights = Path(payload.weights_path)
    settings = request.app.state.settings
    if not weights.is_absolute():
        weights = settings.resolve(payload.weights_path)
    live_camera = getattr(request.app.state, "live_camera", None)
    if live_camera is None:
        raise HTTPException(503, "Live camera not available")
    try:
        job = mgr.start_live(
            weights_path=weights,
            conf=payload.conf,
            iou=payload.iou,
            live_camera=live_camera,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    return _to_live_status(job)


@router.get("/live-infer/{job_id}", response_model=LiveInferStatusResponse)
async def get_live_infer(job_id: str, request: Request) -> LiveInferStatusResponse:
    mgr = _manager(request)
    job = mgr.get_live(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _to_live_status(job)


@router.delete("/live-infer/{job_id}", response_model=LiveInferStatusResponse)
async def cancel_live_infer(job_id: str, request: Request) -> LiveInferStatusResponse:
    mgr = _manager(request)
    job = mgr.get_live(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    mgr.cancel_live(job_id)
    return _to_live_status(job)


@router.post("/infer/{clip_id}", response_model=InferStatusResponse)
async def start_infer_no_run(
    clip_id: str, payload: InferRequest, request: Request
) -> InferStatusResponse:
    """Start inference on a clip without needing a specific run_id.
    Creates a temporary run 'temp_infer' if needed."""
    log.info("start_infer_no_run: clip_id=%s, weights=%s", clip_id, payload.weights_path)
    mgr = _manager(request)
    weights = Path(payload.weights_path)
    settings = request.app.state.settings
    if not weights.is_absolute():
        weights = settings.resolve(payload.weights_path)

    # Use a temp run directory for standalone inference
    run_id = "temp_infer"
    run_dir = request.app.state.store.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        job = mgr.start(
            run_id=run_id,
            clip_id=clip_id,
            weights_path=weights,
            conf=payload.conf,
            iou=payload.iou,
        )
        log.info("start_infer_no_run: created job %s", job.job_id)
    except FileNotFoundError as e:
        log.error("start_infer_no_run: %s", e)
        raise HTTPException(404, str(e)) from e
    return _to_status(job)


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


@router.get("/predictions/{clip_id}", response_model=InferPredictionsResponse)
async def get_predictions_no_run(
    clip_id: str,
    request: Request,
    frame: int | None = None,
    from_idx: int | None = None,
    to_idx: int | None = None,
) -> InferPredictionsResponse:
    """Get predictions from temp_infer run."""
    return await get_predictions("temp_infer", clip_id, request, frame, from_idx, to_idx)


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
