import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class CropAugmentConfig:
    enabled: bool = False
    variants: int = 2
    scale_min: float = 1.05
    scale_max: float = 1.3
    min_iou: float = 0.85


def _clip_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _crop_with_scale(image: np.ndarray, box: Tuple[int, int, int, int], scale: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    nw, nh = bw * scale, bh * scale
    nx1 = _clip_int(cx - nw / 2.0, 0, w - 1)
    ny1 = _clip_int(cy - nh / 2.0, 0, h - 1)
    nx2 = _clip_int(cx + nw / 2.0, nx1 + 1, w)
    ny2 = _clip_int(cy + nh / 2.0, ny1 + 1, h)
    return image[ny1:ny2, nx1:nx2].copy(), (nx1, ny1, nx2, ny2)


def build_crop_variants(
    image: np.ndarray,
    masks: Iterable[np.ndarray],
    config: CropAugmentConfig,
) -> List[np.ndarray]:
    if not config.enabled:
        return []

    candidates: List[np.ndarray] = []
    accepted_boxes: List[Tuple[int, int, int, int]] = []
    masks_list = list(masks)
    if not masks_list:
        return []

    for mask in masks_list:
        base_box = _bbox_from_mask(mask)
        if base_box == (0, 0, 0, 0):
            continue
        for _ in range(max(0, config.variants)):
            scale = random.uniform(config.scale_min, config.scale_max)
            crop, crop_box = _crop_with_scale(image, base_box, scale)
            if crop.size == 0:
                continue
            if any(_iou_xyxy(crop_box, other) >= config.min_iou for other in accepted_boxes):
                continue
            candidates.append(crop)
            accepted_boxes.append(crop_box)
    return candidates
