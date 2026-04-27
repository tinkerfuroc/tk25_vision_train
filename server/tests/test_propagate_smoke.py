"""Native SAM3 video-tracker smoke test.

Synthetic 6-frame "video": 480×640 black background; an 80×80 white square
translates by (10, 0) per frame. We seed the tracker with the square's
mask at frame 0 and assert the propagated mask stays on top of the square
across all 6 frames.

Gated by env var `TK_VISION_GPU_TESTS=1` because it requires CUDA + the
SAM3 checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

CKPT_DIR = Path(__file__).resolve().parents[2] / "sam3_checkpoint_hf"


def _square_image(w: int, h: int, x: int, y: int, side: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y : y + side, x : x + side, :] = 255
    return img


def _square_mask(w: int, h: int, x: int, y: int, side: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y : y + side, x : x + side] = True
    return m


@pytest.mark.skipif(
    os.environ.get("TK_VISION_GPU_TESTS") != "1",
    reason="GPU smoke test gated by TK_VISION_GPU_TESTS=1",
)
@pytest.mark.skipif(
    not (CKPT_DIR / "model.safetensors").is_file(),
    reason=f"sam3 checkpoint not present at {CKPT_DIR}",
)
def test_propagate_translating_square_identity_preserved() -> None:
    import torch  # noqa: F401  — required indirectly

    from tk_vision.annotate.sam3 import Sam3Engine

    h, w = 480, 640
    side = 80
    n_frames = 6
    step = 10
    x0, y0 = 100, 200

    frames = [_square_image(w, h, x0 + step * t, y0, side) for t in range(n_frames)]
    seed = _square_mask(w, h, x0, y0, side)

    eng = Sam3Engine(
        model_dir=str(CKPT_DIR),
        device="cuda",
        dtype="bfloat16",
    )
    eng.load()

    yielded = list(eng.propagate_video(frames=frames, seed_masks={0: seed}))
    assert len(yielded) == n_frames, f"expected {n_frames} frames yielded, got {len(yielded)}"

    seed_area = int(seed.sum())
    for offset, masks in yielded:
        assert 0 in masks, f"track 0 missing at frame {offset}"
        m = masks[0]
        assert m.shape == (h, w)
        ys, xs = np.where(m)
        assert ys.size > 0, f"empty mask at frame {offset}"
        cx, cy = float(xs.mean()), float(ys.mean())
        expected_cx = x0 + step * offset + side / 2.0
        expected_cy = y0 + side / 2.0
        assert abs(cx - expected_cx) < 5.0, f"frame {offset}: cx {cx:.1f} far from {expected_cx:.1f}"
        assert abs(cy - expected_cy) < 5.0, f"frame {offset}: cy {cy:.1f} far from {expected_cy:.1f}"
        area = int(m.sum())
        assert 0.85 * seed_area < area < 1.15 * seed_area, (
            f"frame {offset}: area {area} outside ±15% of seed {seed_area}"
        )
