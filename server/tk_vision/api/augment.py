from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.augment_service import augment_run

router = APIRouter(prefix="/api/runs", tags=["augment"])


class AugmentRequest(BaseModel):
    run_id: str
    multiplier: int | None = Field(None, ge=2)
    seed: int = 0


class AugmentSkipped(BaseModel):
    file: str
    reason: str


class AugmentResponse(BaseModel):
    run_id: str
    source_frames: int
    written_frames: int
    written_polygons: int
    copy_paste_inserts: int
    skipped: list[AugmentSkipped]


@router.post("/{run_id}/augment", response_model=AugmentResponse)
async def augment_dataset(run_id: str, payload: AugmentRequest, request: Request) -> AugmentResponse:
    if payload.run_id != run_id:
        raise HTTPException(400, "run_id mismatch in path vs body")
    settings = request.app.state.settings
    store = request.app.state.store
    run_dir = store.runs_dir / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {run_id}")
    cfg = settings.augment.model_copy()
    if payload.multiplier is not None:
        cfg.multiplier = payload.multiplier
    try:
        stats = await asyncio.to_thread(augment_run, run_dir, cfg, seed=payload.seed)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e
    return AugmentResponse(
        run_id=stats.run_id,
        source_frames=stats.source_frames,
        written_frames=stats.written_frames,
        written_polygons=stats.written_polygons,
        copy_paste_inserts=stats.copy_paste_inserts,
        skipped=[AugmentSkipped(file=f, reason=r) for f, r in stats.skipped],
    )
