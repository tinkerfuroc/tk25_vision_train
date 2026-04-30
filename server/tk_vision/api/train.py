from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.train_service import TERMINAL_STATUSES, TrainJob, TrainStatus as TrainStatusLiteral

log = logging.getLogger("tk_vision.api.train")

router = APIRouter(prefix="/api/runs", tags=["train"])


class ModelInfo(BaseModel):
    """A trained model (best.pt) with metadata."""
    name: str  # Directory name
    path: str  # Relative path to best.pt
    run_id: str | None = None  # Associated export run_id if determinable
    metrics: dict | None = None
    created_at: float | None = None


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    """List all trained models (best.pt files) from the runs directory."""
    settings = request.app.state.settings
    models_dir = settings.resolve(settings.train.project_dir)
    if not models_dir.exists():
        return []

    out: list[ModelInfo] = []
    for run_path in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_path.is_dir():
            continue
        best_pt = run_path / "weights" / "best.pt"
        if not best_pt.exists():
            continue

        # Try to read metrics.json
        metrics = None
        metrics_path = run_path / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception:
                pass

        # Get creation time
        try:
            created_at = best_pt.stat().st_mtime
        except Exception:
            created_at = None

        out.append(ModelInfo(
            name=run_path.name,
            path=str(best_pt.relative_to(settings.resolve("."))),
            run_id=None,  # Can't easily map back to export run_id
            metrics=metrics,
            created_at=created_at,
        ))
    return out


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


class RunSummary(BaseModel):
    """Summary of a run directory on disk (may or may not have active training job)."""
    run_id: str
    run_dir: str
    has_data: bool  # data.yaml exists
    has_model: bool  # best.pt exists
    has_augment: bool  # augment.json exists
    train_frames: int = 0
    val_frames: int = 0
    classes: list[str] = []
    metrics: dict | None = None
    weights_path: str | None = None  # Relative path to best.pt from project root


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


@router.get("", response_model=list[RunSummary])
async def list_runs(request: Request) -> list[RunSummary]:
    """List all run directories with their training status."""
    store = request.app.state.store
    runs_dir = store.runs_dir
    if not runs_dir.exists():
        return []

    out: list[RunSummary] = []
    for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_path.is_dir():
            continue
        run_id = run_path.name
        data_yaml = run_path / "data.yaml"
        best_pt = run_path / "weights" / "best.pt"
        augment_json = run_path / "augment.json"

        # Try to read data.yaml for class/frame counts
        classes: list[str] = []
        train_frames = 0
        val_frames = 0
        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml) as f:
                    data = yaml.safe_load(f) or {}
                classes = data.get("names", [])
                if isinstance(classes, dict):
                    classes = [classes.get(i, str(i)) for i in range(len(classes))]
            except Exception:
                pass
            # Count train/val images
            train_img_dir = run_path / "images" / "train"
            val_img_dir = run_path / "images" / "val"
            if train_img_dir.exists():
                train_frames = len(list(train_img_dir.glob("*.jpg")))
            if val_img_dir.exists():
                val_frames = len(list(val_img_dir.glob("*.jpg")))

        # Try to read metrics.json
        metrics = None
        metrics_files = list(run_path.rglob("metrics.json"))
        if metrics_files:
            try:
                metrics = json.loads(metrics_files[0].read_text())
            except Exception:
                pass

        out.append(RunSummary(
            run_id=run_id,
            run_dir=str(run_path),
            has_data=data_yaml.exists(),
            has_model=best_pt.exists(),
            has_augment=augment_json.exists(),
            train_frames=train_frames,
            val_frames=val_frames,
            classes=classes,
            metrics=metrics,
            weights_path=str(best_pt.relative_to(store.runs_dir.parent)) if best_pt.exists() else None,
        ))
    return out


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(run_id: str, request: Request) -> RunSummary:
    """Get details for a specific run."""
    store = request.app.state.store
    run_path = store.runs_dir / run_id
    if not run_path.exists():
        raise HTTPException(404, f"run not found: {run_id}")

    data_yaml = run_path / "data.yaml"
    best_pt = run_path / "weights" / "best.pt"
    augment_json = run_path / "augment.json"

    classes: list[str] = []
    train_frames = 0
    val_frames = 0
    if data_yaml.exists():
        try:
            import yaml
            with open(data_yaml) as f:
                data = yaml.safe_load(f) or {}
            classes = data.get("names", [])
            if isinstance(classes, dict):
                classes = [classes.get(i, str(i)) for i in range(len(classes))]
        except Exception:
            pass
        train_img_dir = run_path / "images" / "train"
        val_img_dir = run_path / "images" / "val"
        if train_img_dir.exists():
            train_frames = len(list(train_img_dir.glob("*.jpg")))
        if val_img_dir.exists():
            val_frames = len(list(val_img_dir.glob("*.jpg")))

    metrics = None
    metrics_files = list(run_path.rglob("metrics.json"))
    if metrics_files:
        try:
            metrics = json.loads(metrics_files[0].read_text())
        except Exception:
            pass

    return RunSummary(
        run_id=run_id,
        run_dir=str(run_path),
        has_data=data_yaml.exists(),
        has_model=best_pt.exists(),
        has_augment=augment_json.exists(),
        train_frames=train_frames,
        val_frames=val_frames,
        classes=classes,
        metrics=metrics,
        weights_path=str(best_pt.relative_to(store.runs_dir.parent)) if best_pt.exists() else None,
    )


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
