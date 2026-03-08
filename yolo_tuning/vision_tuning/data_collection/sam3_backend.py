import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import supervision as sv
import torch


@dataclass
class SegmentationBatch:
    detections: sv.Detections
    metadata: Dict[str, Any]


class SegmentationBackend(ABC):
    @abstractmethod
    def segment(self, image_bgr: np.ndarray, prompts: Iterable[str]) -> SegmentationBatch:
        raise NotImplementedError

    @abstractmethod
    def segment_with_points(self, image_bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def segment_with_box(self, image_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OfficialSAM3Backend(SegmentationBackend):
    """Adapter around a SAM3 predictor implementation.

    This class assumes a SAM3 runtime exposing a predictor-like object with:
    - set_image(rgb_image)
    - predict(point_coords=..., point_labels=..., multimask_output=False)
    - predict(box=..., multimask_output=False)
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.predictor = self._build_predictor(checkpoint_path, self.device)

    def _build_predictor(self, checkpoint_path: str, device: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        # Official backend: require an externally installed SAM3 package.
        # We intentionally do not vendor SAM3 into this repository.
        try:
            import sam3  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Official SAM3 runtime not available. Install the SAM3 package required by your environment."
            ) from exc

        if hasattr(sam3, "build_predictor"):
            return sam3.build_predictor(checkpoint_path=checkpoint_path, device=device)
        if hasattr(sam3, "SAM3Predictor"):
            return sam3.SAM3Predictor(checkpoint_path=checkpoint_path, device=device)

        raise RuntimeError("Unsupported SAM3 package shape. Expected build_predictor() or SAM3Predictor class.")

    def _ensure_mask(self, mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(image_shape, dtype=bool)
        if mask.dtype != np.bool_:
            return (mask > 0.5).astype(bool)
        return mask

    def segment(self, image_bgr: np.ndarray, prompts: Iterable[str]) -> SegmentationBatch:
        # SAM3 runtimes are not guaranteed to support text prompts directly.
        # MVP behavior: generate masks from automatic mode when available, and attach prompt metadata.
        rgb_image = image_bgr[:, :, ::-1]
        if not hasattr(self.predictor, "predict_auto"):
            return SegmentationBatch(detections=sv.Detections.empty(), metadata={"prompts": list(prompts)})

        auto_output = self.predictor.predict_auto(rgb_image)
        masks = np.array(auto_output.get("masks", []), dtype=bool)
        scores = np.array(auto_output.get("scores", []), dtype=float) if len(masks) else np.array([])
        if len(masks) == 0:
            return SegmentationBatch(detections=sv.Detections.empty(), metadata={"prompts": list(prompts)})

        boxes = sv.mask_to_xyxy(masks=masks)
        class_ids = np.zeros(len(masks), dtype=int)
        detections = sv.Detections(xyxy=boxes, mask=masks, confidence=scores, class_id=class_ids)
        return SegmentationBatch(detections=detections, metadata={"prompts": list(prompts)})

    def segment_with_points(self, image_bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        rgb_image = image_bgr[:, :, ::-1]
        self.predictor.set_image(rgb_image)
        masks, _, _ = self.predictor.predict(
            point_coords=np.asarray(points_xy),
            point_labels=np.ones(len(points_xy), dtype=np.int32),
            multimask_output=False,
        )
        return self._ensure_mask(masks[0] if masks is not None and len(masks) else None, image_bgr.shape[:2])

    def segment_with_box(self, image_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        rgb_image = image_bgr[:, :, ::-1]
        self.predictor.set_image(rgb_image)
        masks, _, _ = self.predictor.predict(box=np.asarray(box_xyxy), multimask_output=False)
        return self._ensure_mask(masks[0] if masks is not None and len(masks) else None, image_bgr.shape[:2])
