from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tk_vision.annotate.sam3 import Sam3Engine, is_degenerate_mask
from tk_vision.data.manifest import Clip, ClipMeta, Mask, Track
from tk_vision.data.persistence import (
    ProjectStore,
    mask_path_for,
    read_mask,
    write_mask,
)
from tk_vision.services.label_service import cleanup_clip


def test_is_degenerate_full_frame() -> None:
    m = np.ones((480, 640), dtype=bool)
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64) is True


def test_is_degenerate_edge_touching_under_area_cap() -> None:
    """40% area mask with 1-pixel border on all 4 edges is rejected."""
    m = np.zeros((480, 640), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    # Plus a larger interior blob so area is non-trivial but under 45%.
    m[100:300, 100:400] = True
    area_frac = float(m.sum()) / m.size
    assert 0.10 < area_frac < 0.45
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64) is True


def test_is_degenerate_legitimate_mask() -> None:
    m = np.zeros((480, 640), dtype=bool)
    m[100:200, 200:340] = True
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64) is False


def test_is_degenerate_too_small() -> None:
    m = np.zeros((480, 640), dtype=bool)
    m[10:14, 10:14] = True  # 16 pixels < min 64
    assert is_degenerate_mask(m, max_area_frac=0.45, min_area_px=64) is True


def _seed_clip_with_masks(tmp_path: Path, masks_by_track: dict[tuple[int, str], np.ndarray]) -> ProjectStore:
    """Build a project with one clip and prepopulated tracks/masks."""
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
        frame_count=1,
        created_at=0.0,
        intrinsics=None,
    )
    store.write_meta(meta)
    clip = Clip(**meta.model_dump())
    for (tid, seeded_from), mask in masks_by_track.items():
        track = Track(
            track_id=tid,
            class_id=0,
            label="test",
            seeded_from=seeded_from,  # type: ignore[arg-type]
        )
        p = mask_path_for(cdir, tid, 0)
        write_mask(p, mask)
        track.masks[0] = Mask(
            frame_idx=0,
            track_id=tid,
            rle_path=str(p.relative_to(cdir)),
            source="text" if seeded_from == "text" else "click",
        )
        clip.tracks.append(track)
    store.write_clip(clip)
    return store


def test_cleanup_drops_full_frame_keeps_small(tmp_path: Path) -> None:
    full = np.ones((480, 640), dtype=bool)
    small = np.zeros((480, 640), dtype=bool)
    small[100:200, 200:340] = True
    store = _seed_clip_with_masks(
        tmp_path,
        {
            (0, "text"): full,
            (1, "text"): small,
            (2, "text"): full,
        },
    )
    clip, dropped = cleanup_clip(store, "test_clip")
    assert sorted(dropped) == [0, 2]
    assert [t.track_id for t in clip.tracks] == [1]
    # disk: dropped track dirs gone, kept track survives
    cdir = store.clip_dir("test_clip")
    assert not (cdir / "masks" / "0").exists()
    assert not (cdir / "masks" / "2").exists()
    assert (cdir / "masks" / "1").exists()


def test_cleanup_spares_click_tracks(tmp_path: Path) -> None:
    """User-edited tracks survive even if their mask happens to be degenerate."""
    full = np.ones((480, 640), dtype=bool)
    store = _seed_clip_with_masks(
        tmp_path,
        {
            (0, "text"): full,
            (1, "click"): full,  # user-owned; spared
        },
    )
    clip, dropped = cleanup_clip(store, "test_clip")
    assert dropped == [0]
    assert [t.track_id for t in clip.tracks] == [1]


def test_cleanup_no_op_on_clean_clip(tmp_path: Path) -> None:
    small = np.zeros((480, 640), dtype=bool)
    small[100:200, 200:340] = True
    store = _seed_clip_with_masks(tmp_path, {(0, "text"): small})
    _, dropped = cleanup_clip(store, "test_clip")
    assert dropped == []
