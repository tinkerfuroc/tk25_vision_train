from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.export_service import ExportStats, export_run

router = APIRouter(prefix="/api", tags=["export"])


class ExportRequest(BaseModel):
    run_id: str
    clip_ids: list[str] | None = None
    train_ratio: float = Field(0.85, gt=0.0, lt=1.0)
    per_clip_split: bool = True
    seed: int = 0
    overwrite: bool = False


class ExportClipRequest(BaseModel):
    run_id: str
    train_ratio: float = Field(0.85, gt=0.0, lt=1.0)
    per_clip_split: bool = Field(
        False, description="Single clip → per-clip-split would put all frames on one side."
    )
    seed: int = 0
    overwrite: bool = False


class SkippedFrame(BaseModel):
    clip_id: str
    frame_idx: int
    reason: str


class ExportResponse(BaseModel):
    run_id: str
    run_dir: str
    classes: list[str]
    train_frames: int
    val_frames: int
    train_polygons: int
    val_polygons: int
    skipped_frames: list[SkippedFrame]


def _stats_to_response(stats: ExportStats, run_dir: str) -> ExportResponse:
    return ExportResponse(
        run_id=stats.run_id,
        run_dir=run_dir,
        classes=stats.classes,
        train_frames=stats.train_frames,
        val_frames=stats.val_frames,
        train_polygons=stats.train_polygons,
        val_polygons=stats.val_polygons,
        skipped_frames=[
            SkippedFrame(clip_id=s.clip_id, frame_idx=s.frame_idx, reason=s.reason)
            for s in stats.skipped_frames
        ],
    )


async def _do_export(request: Request, **kwargs) -> ExportResponse:
    store = request.app.state.store
    settings = request.app.state.settings
    try:
        stats = await asyncio.to_thread(export_run, store, settings=settings, **kwargs)
    except FileExistsError as e:
        raise HTTPException(409, str(e)) from e
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e
    return _stats_to_response(stats, str(store.runs_dir / kwargs["run_id"]))


@router.post("/runs/export", response_model=ExportResponse)
async def export_dataset(payload: ExportRequest, request: Request) -> ExportResponse:
    return await _do_export(
        request,
        run_id=payload.run_id,
        clip_ids=payload.clip_ids,
        train_ratio=payload.train_ratio,
        per_clip_split=payload.per_clip_split,
        seed=payload.seed,
        overwrite=payload.overwrite,
    )


@router.post("/clips/{clip_id}/export", response_model=ExportResponse)
async def export_single_clip(
    clip_id: str, payload: ExportClipRequest, request: Request
) -> ExportResponse:
    store = request.app.state.store
    if not store.clip_dir(clip_id).exists():
        raise HTTPException(404, "Clip not found")
    return await _do_export(
        request,
        run_id=payload.run_id,
        clip_ids=[clip_id],
        train_ratio=payload.train_ratio,
        per_clip_split=payload.per_clip_split,
        seed=payload.seed,
        overwrite=payload.overwrite,
    )
