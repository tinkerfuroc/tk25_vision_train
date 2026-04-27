"""Tests for frame-level review/edit operations: delete, restore, prune."""

from __future__ import annotations

from pathlib import Path

import pytest

from tk_vision.config import Settings
from tk_vision.data.manifest import Clip, ClipMeta
from tk_vision.data.persistence import ProjectStore
from tk_vision.services.label_service import LabelService


def _make_service(tmp_path: Path, frame_count: int = 100) -> tuple[LabelService, str]:
    store = ProjectStore(tmp_path)
    clip_id = "test_clip"
    cdir = store.clip_dir(clip_id)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "frames").mkdir(exist_ok=True)
    meta = ClipMeta(
        clip_id=clip_id,
        source="folder",
        bag_path=None,
        folder_path=None,
        fps=30.0,
        width=640,
        height=480,
        frame_count=frame_count,
        created_at=0.0,
        intrinsics=None,
    )
    store.write_meta(meta)
    store.write_clip(Clip(**meta.model_dump()))

    settings = Settings()
    settings.repo_root = str(tmp_path)
    svc = LabelService(store, sam3=None, settings=settings)  # type: ignore[arg-type]
    return svc, clip_id


def test_delete_then_restore_frame(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path)
    clip = svc.delete_frame(cid, 5)
    assert 5 in clip.deleted_frames
    clip = svc.restore_frame(cid, 5)
    assert 5 not in clip.deleted_frames


def test_delete_frame_idempotent(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path)
    svc.delete_frame(cid, 7)
    clip = svc.delete_frame(cid, 7)
    assert clip.deleted_frames.count(7) == 1


def test_restore_frame_no_op_when_not_deleted(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path)
    clip = svc.restore_frame(cid, 12)
    assert clip.deleted_frames == []


def test_prune_marks_tail_deleted(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path, frame_count=10)
    clip = svc.prune_frames(cid, from_idx=4)
    assert clip.deleted_frames == [4, 5, 6, 7, 8, 9]


def test_prune_unions_with_existing_deletions(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path, frame_count=10)
    svc.delete_frame(cid, 1)
    clip = svc.prune_frames(cid, from_idx=7)
    assert clip.deleted_frames == [1, 7, 8, 9]


def test_prune_rejects_out_of_range(tmp_path: Path) -> None:
    svc, cid = _make_service(tmp_path, frame_count=10)
    with pytest.raises(ValueError):
        svc.prune_frames(cid, from_idx=10)
    with pytest.raises(ValueError):
        svc.prune_frames(cid, from_idx=-1)
