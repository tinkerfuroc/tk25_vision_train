"""Tests for augment_service."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tk_vision.config import AugmentCfg
from tk_vision.services.augment_service import (
    _load_label_file,
    _mask_to_polygons,
    _polygon_to_mask,
    augment_run,
)


def _seed_export(run_dir: Path, *, n_train: int = 3, n_val: int = 1, h: int = 96, w: int = 128) -> None:
    for split, n in (("train", n_train), ("val", n_val)):
        (run_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (run_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(run_dir / "images" / split / f"clip_{i:06d}.jpg"), img)
            poly = [10 / w, 10 / h, 60 / w, 10 / h, 60 / w, 50 / h, 10 / w, 50 / h]
            (run_dir / "labels" / split / f"clip_{i:06d}.txt").write_text(
                f"0 {' '.join(f'{v:.6f}' for v in poly)}\n"
            )


def _augment_cfg(**overrides) -> AugmentCfg:
    payload = {
        "multiplier": 3,
        "apply_to": "train_only",
        "ops": [
            {"name": "HorizontalFlip", "p": 1.0},
            {"name": "RandomBrightnessContrast", "p": 1.0, "brightness_limit": 0.1},
        ],
        "copy_paste": {"enabled": False},
    }
    payload.update(overrides)
    return AugmentCfg(**payload)


def test_polygon_mask_roundtrip() -> None:
    h, w = 96, 128
    poly = [0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]
    mask = _polygon_to_mask(poly, h, w)
    assert mask.sum() > 0
    polys = _mask_to_polygons(mask)
    assert len(polys) == 1
    assert len(polys[0]) >= 8


def test_load_label_file_skips_blank_and_short(tmp_path: Path) -> None:
    p = tmp_path / "lbl.txt"
    p.write_text("\n0 0.1 0.1 0.2 0.2\n0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n\n")
    rows = _load_label_file(p)
    assert len(rows) == 1


def test_augment_writes_multiplier_copies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    _seed_export(run_dir, n_train=2)
    stats = augment_run(run_dir, _augment_cfg(multiplier=3), seed=0)
    assert stats.source_frames == 2
    assert stats.written_frames == 6  # 2 * 3
    train_imgs = sorted((run_dir / "images" / "train").glob("*.jpg"))
    assert len(train_imgs) == 6
    for img in train_imgs:
        lbl = run_dir / "labels" / "train" / (img.stem + ".txt")
        assert lbl.exists()
        rows = _load_label_file(lbl)
        assert rows, f"empty label for {img}"


def test_augment_skips_val_when_train_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run2"
    _seed_export(run_dir, n_train=1, n_val=2)
    augment_run(run_dir, _augment_cfg(apply_to="train_only"), seed=0)
    val_imgs = sorted((run_dir / "images" / "val").glob("*.jpg"))
    assert len(val_imgs) == 2  # untouched
    val_lbls = sorted((run_dir / "labels" / "val").glob("*.txt"))
    assert len(val_lbls) == 2


def test_augment_idempotent_skips_already_augmented(tmp_path: Path) -> None:
    run_dir = tmp_path / "run3"
    _seed_export(run_dir, n_train=2)
    augment_run(run_dir, _augment_cfg(multiplier=3), seed=0)
    n_after_first = len(list((run_dir / "images" / "train").glob("*.jpg")))
    augment_run(run_dir, _augment_cfg(multiplier=3), seed=0)
    n_after_second = len(list((run_dir / "images" / "train").glob("*.jpg")))
    assert n_after_first == n_after_second == 6


def test_augment_rejects_multiplier_le_one(tmp_path: Path) -> None:
    run_dir = tmp_path / "run4"
    _seed_export(run_dir, n_train=1)
    with pytest.raises(ValueError):
        augment_run(run_dir, _augment_cfg(multiplier=1), seed=0)


def test_augment_missing_run_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        augment_run(tmp_path / "nope", _augment_cfg(), seed=0)


def test_augment_copy_paste_increments_counter(tmp_path: Path) -> None:
    run_dir = tmp_path / "run5"
    _seed_export(run_dir, n_train=4)
    cfg = _augment_cfg(
        multiplier=3,
        copy_paste={"enabled": True, "p": 1.0, "min_area_frac": 0.001},
    )
    stats = augment_run(run_dir, cfg, seed=0)
    # multiplier=3 → variant 0 is original (no paste), 1+2 each roll p=1.0.
    assert stats.copy_paste_inserts >= 1
