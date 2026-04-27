"""Tests for YOLO-seg export service."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tk_vision.config import Settings
from tk_vision.data.manifest import Clip, ClipMeta, Mask, Track
from tk_vision.data.persistence import ProjectStore, mask_path_for, write_mask
from tk_vision.services.export_service import (
    _mask_to_polygons,
    _normalize_polygon,
    export_run,
)


def _build_clip(
    store: ProjectStore,
    *,
    clip_id: str,
    frame_count: int = 4,
    width: int = 64,
    height: int = 48,
    tracks: list[tuple[int, int, str, dict[int, np.ndarray]]] | None = None,
) -> Clip:
    """Helper: create a clip with frames + masks on disk."""
    cdir = store.clip_dir(clip_id)
    fdir = cdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    img = (np.random.default_rng(0).integers(0, 256, size=(height, width, 3))).astype(np.uint8)
    for i in range(frame_count):
        cv2.imwrite(str(fdir / f"{i:06d}.jpg"), img)

    meta = ClipMeta(
        clip_id=clip_id,
        source="folder",
        bag_path=None,
        folder_path=None,
        fps=30.0,
        width=width,
        height=height,
        frame_count=frame_count,
        created_at=0.0,
        intrinsics=None,
    )
    store.write_meta(meta)

    track_models: list[Track] = []
    for tid, cid, label, masks_by_idx in tracks or []:
        mask_models = {}
        for fi, m in masks_by_idx.items():
            p = mask_path_for(cdir, tid, fi)
            write_mask(p, m)
            mask_models[fi] = Mask(
                frame_idx=fi,
                track_id=tid,
                rle_path=str(p),
                source="text",
            )
        track_models.append(
            Track(track_id=tid, class_id=cid, label=label, seeded_from="text", masks=mask_models)
        )
    clip = Clip(**meta.model_dump(), tracks=track_models)
    store.write_clip(clip)
    return clip


def _rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _disconnected_two_blob_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[5:15, 5:15] = True
    m[25:40, 30:50] = True  # second blob, no overlap
    return m


def test_mask_to_polygons_multi_contour() -> None:
    m = _disconnected_two_blob_mask(48, 64)
    polys = _mask_to_polygons(m)
    assert len(polys) == 2
    for p in polys:
        assert len(p) >= 6  # at least 3 points (x,y pairs)
        assert len(p) % 2 == 0


def test_mask_to_polygons_drops_speck() -> None:
    m = np.zeros((48, 64), dtype=bool)
    m[1, 1] = True  # 1-pixel speck
    m[10:30, 10:30] = True
    polys = _mask_to_polygons(m, min_area_px=4.0)
    assert len(polys) == 1


def test_normalize_polygon_pairs_xy() -> None:
    out = _normalize_polygon([10.0, 20.0, 30.0, 40.0], 100, 200)
    assert out == [0.1, 0.1, 0.3, 0.2]


def test_export_per_clip_split(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    for cid in ("clipA", "clipB"):
        _build_clip(
            store,
            clip_id=cid,
            frame_count=3,
            width=w,
            height=h,
            tracks=[
                (0, 0, "thing", {0: _rect_mask(h, w, 5, 5, 25, 25),
                                  1: _rect_mask(h, w, 6, 6, 26, 26),
                                  2: _rect_mask(h, w, 7, 7, 27, 27)}),
            ],
        )
    settings = Settings()
    settings.repo_root = str(tmp_path)
    stats = export_run(
        store,
        run_id="r1",
        train_ratio=0.5,
        per_clip_split=True,
        seed=0,
        settings=settings,
    )
    assert stats.train_frames + stats.val_frames == 6
    run_dir = store.runs_dir / "r1"
    assert (run_dir / "data.yaml").exists()
    assert (run_dir / "export.json").exists()
    train_imgs = sorted((run_dir / "images" / "train").glob("*.jpg"))
    val_imgs = sorted((run_dir / "images" / "val").glob("*.jpg"))
    train_clips = {p.name.split("__")[0] for p in train_imgs}
    val_clips = {p.name.split("__")[0] for p in val_imgs}
    assert train_clips.isdisjoint(val_clips)
    for img in train_imgs + val_imgs:
        split = img.parent.name
        lbl = run_dir / "labels" / split / (img.stem + ".txt")
        assert lbl.exists()
        text = lbl.read_text().strip()
        parts = text.split()
        assert int(parts[0]) == 0


def test_export_writes_one_polygon_per_contour(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    _build_clip(
        store,
        clip_id="cA",
        frame_count=2,
        width=w,
        height=h,
        tracks=[
            (0, 0, "two", {0: _disconnected_two_blob_mask(h, w),
                           1: _disconnected_two_blob_mask(h, w)}),
        ],
    )
    settings = Settings()
    settings.repo_root = str(tmp_path)
    stats = export_run(
        store, run_id="r2", train_ratio=0.5, per_clip_split=False, seed=0, settings=settings
    )
    run_dir = store.runs_dir / "r2"
    for split in ("train", "val"):
        for lbl in (run_dir / "labels" / split).glob("*.txt"):
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            assert len(lines) == 2, f"{lbl}: expected 2 polygons, got {len(lines)}"
    assert stats.train_polygons + stats.val_polygons == 4


def test_export_skips_deleted_frames(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    clip = _build_clip(
        store,
        clip_id="cA",
        frame_count=4,
        width=w,
        height=h,
        tracks=[
            (0, 0, "thing", {fi: _rect_mask(h, w, 5, 5, 20, 20) for fi in range(4)}),
        ],
    )
    clip.deleted_frames = [1, 3]
    store.write_clip(clip)
    settings = Settings()
    settings.repo_root = str(tmp_path)
    stats = export_run(
        store, run_id="r3", train_ratio=0.5, per_clip_split=False, seed=0, settings=settings
    )
    assert stats.train_frames + stats.val_frames == 2
    run_dir = store.runs_dir / "r3"
    all_stems = sorted(
        p.stem for p in list((run_dir / "images" / "train").glob("*.jpg"))
        + list((run_dir / "images" / "val").glob("*.jpg"))
    )
    indices = sorted(int(s.rsplit("__", 1)[1]) for s in all_stems)
    assert indices == [0, 2]


def test_export_overwrite_required(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    _build_clip(
        store, clip_id="cA", frame_count=2, width=w, height=h,
        tracks=[(0, 0, "x", {0: _rect_mask(h, w, 5, 5, 20, 20),
                              1: _rect_mask(h, w, 6, 6, 21, 21)})],
    )
    settings = Settings()
    settings.repo_root = str(tmp_path)
    export_run(store, run_id="r4", train_ratio=0.5, per_clip_split=False, seed=0, settings=settings)
    with pytest.raises(FileExistsError):
        export_run(store, run_id="r4", train_ratio=0.5, per_clip_split=False, seed=0, settings=settings)
    export_run(
        store, run_id="r4", train_ratio=0.5, per_clip_split=False,
        seed=0, overwrite=True, settings=settings,
    )


def test_export_rejects_invalid_run_id(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    settings = Settings()
    settings.repo_root = str(tmp_path)
    with pytest.raises(ValueError):
        export_run(store, run_id="../bad", settings=settings)
    with pytest.raises(ValueError):
        export_run(store, run_id="x/y", settings=settings)


def test_export_rejects_no_labeled_frames(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    _build_clip(store, clip_id="cA", frame_count=2, width=w, height=h, tracks=[])
    settings = Settings()
    settings.repo_root = str(tmp_path)
    with pytest.raises(ValueError):
        export_run(store, run_id="r5", settings=settings)


def test_export_meta_records_classes_and_seed(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    h, w = 48, 64
    _build_clip(
        store, clip_id="cA", frame_count=2, width=w, height=h,
        tracks=[(0, 0, "thing", {0: _rect_mask(h, w, 5, 5, 25, 25),
                                  1: _rect_mask(h, w, 6, 6, 26, 26)})],
    )
    settings = Settings()
    settings.repo_root = str(tmp_path)
    export_run(
        store, run_id="r6", train_ratio=0.5, per_clip_split=False,
        seed=42, settings=settings,
    )
    meta = json.loads((store.runs_dir / "r6" / "export.json").read_text())
    assert meta["seed"] == 42
    assert meta["classes"] == ["thing"]
    assert meta["train_frames"] + meta["val_frames"] == 2
