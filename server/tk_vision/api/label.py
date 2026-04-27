from __future__ import annotations

import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ..data.persistence import mask_path_for, read_mask
from ..services.label_service import cleanup_clip as _cleanup_clip

router = APIRouter(prefix="/api/clips", tags=["label"])


class TrackOut(BaseModel):
    track_id: int
    class_id: int
    label: str
    seeded_from: str
    frame_indices: list[int]


class ClipDetail(BaseModel):
    clip_id: str
    width: int
    height: int
    fps: float
    frame_count: int
    deleted_frames: list[int]
    tracks: list[TrackOut]


def _label_service(request: Request):
    svc = getattr(request.app.state, "label", None)
    if svc is None:
        raise HTTPException(503, "Label service not initialized (SAM3 disabled?)")
    return svc


def _require_clip(request: Request, clip_id: str):
    store = request.app.state.store
    if not store.clip_dir(clip_id).exists():
        raise HTTPException(404, "Clip not found")
    return store


def _clip_detail(clip) -> ClipDetail:
    return ClipDetail(
        clip_id=clip.clip_id,
        width=clip.width,
        height=clip.height,
        fps=clip.fps,
        frame_count=clip.frame_count,
        deleted_frames=list(clip.deleted_frames),
        tracks=[
            TrackOut(
                track_id=t.track_id,
                class_id=t.class_id,
                label=t.label,
                seeded_from=t.seeded_from,
                frame_indices=sorted(t.masks.keys()),
            )
            for t in clip.tracks
        ],
    )


@router.get("/{clip_id}/detail", response_model=ClipDetail)
async def get_clip_detail(clip_id: str, request: Request) -> ClipDetail:
    store = _require_clip(request, clip_id)
    return _clip_detail(store.read_clip(clip_id))


@router.post("/{clip_id}/seed", response_model=ClipDetail)
async def seed(clip_id: str, request: Request) -> ClipDetail:
    svc = _label_service(request)
    _require_clip(request, clip_id)
    return _clip_detail(await svc.seed_first_frame(clip_id))


class RefineRequest(BaseModel):
    track_id: int | None = None
    class_id: int | None = None
    label: str | None = None
    points: list[tuple[int, int, int]] | None = None
    box: tuple[int, int, int, int] | None = None


class RefineResponse(BaseModel):
    track_id: int
    frame_idx: int
    width: int
    height: int


@router.post("/{clip_id}/frames/{idx}/refine", response_model=RefineResponse)
async def refine(
    clip_id: str, idx: int, payload: RefineRequest, request: Request
) -> RefineResponse:
    svc = _label_service(request)
    _require_clip(request, clip_id)
    try:
        clip, track, _mask = await svc.refine_frame(
            clip_id,
            idx,
            track_id=payload.track_id,
            points=payload.points,
            box=payload.box,
            class_id=payload.class_id,
            label=payload.label,
        )
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e
    return RefineResponse(
        track_id=track.track_id,
        frame_idx=idx,
        width=clip.width,
        height=clip.height,
    )


@router.delete("/{clip_id}/tracks/{track_id}", response_model=ClipDetail)
async def delete_track(clip_id: str, track_id: int, request: Request) -> ClipDetail:
    svc = _label_service(request)
    return _clip_detail(svc.delete_track(clip_id, track_id))


class TrackPatch(BaseModel):
    class_id: int | None = None
    label: str | None = None


@router.patch("/{clip_id}/tracks/{track_id}", response_model=ClipDetail)
async def patch_track(
    clip_id: str, track_id: int, payload: TrackPatch, request: Request
) -> ClipDetail:
    svc = _label_service(request)
    try:
        return _clip_detail(svc.update_track(clip_id, track_id, class_id=payload.class_id, label=payload.label))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/{clip_id}/frames/{idx}", response_model=ClipDetail)
async def delete_frame(clip_id: str, idx: int, request: Request) -> ClipDetail:
    svc = _label_service(request)
    return _clip_detail(svc.delete_frame(clip_id, idx))


@router.post("/{clip_id}/frames/{idx}/restore", response_model=ClipDetail)
async def restore_frame(clip_id: str, idx: int, request: Request) -> ClipDetail:
    svc = _label_service(request)
    return _clip_detail(svc.restore_frame(clip_id, idx))


class PruneRequest(BaseModel):
    from_idx: int


@router.post("/{clip_id}/frames/prune", response_model=ClipDetail)
async def prune_frames(
    clip_id: str, payload: PruneRequest, request: Request
) -> ClipDetail:
    svc = _label_service(request)
    try:
        return _clip_detail(svc.prune_frames(clip_id, payload.from_idx))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class CleanupResponse(BaseModel):
    detail: ClipDetail
    dropped_track_ids: list[int]


@router.post("/{clip_id}/cleanup", response_model=CleanupResponse)
async def cleanup_clip(clip_id: str, request: Request) -> CleanupResponse:
    store = _require_clip(request, clip_id)
    clip, dropped = await asyncio.to_thread(_cleanup_clip, store, clip_id)
    return CleanupResponse(detail=_clip_detail(clip), dropped_track_ids=dropped)


def _encode_mask_png(mask_path) -> bytes:
    """Read an RLE mask from disk and encode it as a 4-channel RGBA PNG
    with the mask in the alpha channel and RGB zero.

    The SPA renders each track as a `<div>` with `backgroundColor` set to
    the class colour, clipped via CSS `mask-image` + `mask-mode: alpha`.
    A grayscale PNG (no alpha channel) leaves the implicit alpha at 1.0
    everywhere → background colour leaks across the whole frame. Carrying
    the mask in a real alpha channel is what `mask-mode: alpha` expects.
    """
    mask = read_mask(mask_path)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask.astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("Failed to encode mask")
    return buf.tobytes()


@router.get("/{clip_id}/tracks/{track_id}/masks/{idx}.png")
async def mask_png(clip_id: str, track_id: int, idx: int, request: Request) -> Response:
    store = request.app.state.store
    p = mask_path_for(store.clip_dir(clip_id), track_id, idx)
    try:
        png = await asyncio.to_thread(_encode_mask_png, p)
    except FileNotFoundError as e:
        raise HTTPException(404, "Mask not found") from e
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    return Response(content=png, media_type="image/png")


class TrackAtResponse(BaseModel):
    track_id: int | None
    class_id: int | None
    label: str | None


def _topmost_track(clip, cdir, idx: int, x: int, y: int) -> TrackAtResponse:
    if x < 0 or y < 0 or x >= clip.width or y >= clip.height:
        return TrackAtResponse(track_id=None, class_id=None, label=None)
    # Higher track_id wins (most recently created/edited).
    for track in sorted(clip.tracks, key=lambda t: -t.track_id):
        if idx not in track.masks:
            continue
        try:
            mask = read_mask(mask_path_for(cdir, track.track_id, idx))
        except FileNotFoundError:
            continue
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x]):
            return TrackAtResponse(
                track_id=track.track_id, class_id=track.class_id, label=track.label
            )
    return TrackAtResponse(track_id=None, class_id=None, label=None)


@router.get("/{clip_id}/frames/{idx}/track-at", response_model=TrackAtResponse)
async def track_at_pixel(
    clip_id: str, idx: int, x: int, y: int, request: Request
) -> TrackAtResponse:
    store = _require_clip(request, clip_id)
    clip = store.read_clip(clip_id)
    return await asyncio.to_thread(
        _topmost_track, clip, store.clip_dir(clip_id), idx, x, y
    )
