"""Mask ↔ polygon helpers shared by export and augment services.

YOLO-seg polygon convention: pixel-coord polygons are flat lists
[x0, y0, x1, y1, ...]; normalized polygons divide x by W and y by H.
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygons(
    mask: np.ndarray, *, min_points: int = 3, min_area_px: float = 4.0
) -> list[list[float]]:
    """Convert a boolean/uint8 mask to one polygon per connected component.

    Polygons below `min_area_px` or with fewer than `min_points` vertices
    are dropped. Returns flat [x0,y0,...] lists in pixel coordinates.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[float]] = []
    for c in contours:
        if c.shape[0] < min_points:
            continue
        if cv2.contourArea(c) < min_area_px:
            continue
        out.append(c.reshape(-1, 2).astype(float).flatten().tolist())
    return out


def normalize_polygon(poly: list[float], w: int, h: int) -> list[float]:
    return [v / w if (i % 2 == 0) else v / h for i, v in enumerate(poly)]


def polygon_to_mask(coords: list[float], h: int, w: int) -> np.ndarray:
    """Rasterize a normalized [x0,y0,x1,y1,...] polygon to a uint8 mask."""
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], color=1)
    return mask
