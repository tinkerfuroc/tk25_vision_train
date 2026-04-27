from __future__ import annotations

from pathlib import Path

import numpy as np

from tk_vision.data.persistence import (
    decode_mask_rle,
    encode_mask_rle,
    mask_path_for,
    read_mask,
    write_mask,
)


def test_rle_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    h, w = 96, 128
    blob = rng.integers(0, 2, size=(h, w), dtype=np.int8).astype(bool)
    blob[10:30, 20:60] = True
    blob[50:80, 80:110] = False

    (size, counts) = encode_mask_rle(blob)
    assert size == (h, w)
    out = decode_mask_rle(size, counts)
    assert out.shape == blob.shape
    assert out.dtype == np.bool_
    assert np.array_equal(out, blob)


def test_write_read_mask(tmp_path: Path) -> None:
    blob = np.zeros((48, 64), dtype=bool)
    blob[10:30, 20:50] = True
    p = mask_path_for(tmp_path, track_id=3, frame_idx=42)
    write_mask(p, blob)
    assert p.exists()
    out = read_mask(p)
    assert np.array_equal(out, blob)


def test_mask_png_is_rgba_with_alpha_channel(tmp_path: Path) -> None:
    """The SPA's `mask-mode: alpha` requires a real alpha channel; if the
    server emits luminance-only PNGs the background colour leaks across
    the whole frame. Lock in the RGBA layout."""
    import cv2

    from tk_vision.api.label import _encode_mask_png

    blob = np.zeros((48, 64), dtype=bool)
    blob[10:30, 20:50] = True
    p = mask_path_for(tmp_path, track_id=0, frame_idx=0)
    write_mask(p, blob)

    png = _encode_mask_png(p)
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (48, 64, 4)
    assert decoded.dtype == np.uint8
    # cv2 returns BGRA; alpha is channel 3.
    assert np.array_equal(decoded[..., 3], blob.astype(np.uint8) * 255)
    # RGB band is all zero.
    assert int(decoded[..., :3].sum()) == 0
