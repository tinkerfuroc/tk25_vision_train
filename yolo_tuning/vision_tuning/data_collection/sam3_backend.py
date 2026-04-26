import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image


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
    """Adapter around SAM3 using Sam3Processor for text-based segmentation.

    Uses the official SAM3 package from facebookresearch/sam3 with:
    - Sam3Processor for text-based segmentation
    - SAM3InteractiveImagePredictor for point/box prompts
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.processor, self.predictor = self._build_backends(checkpoint_path, self.device)

    def _build_backends(self, checkpoint_path: str, device: str):
        """Build both text-based processor and interactive predictor."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

        try:
            import sam3  # type: ignore
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(
                "Official SAM3 runtime not available. Install from: pip install git+https://github.com/facebookresearch/sam3.git"
            ) from exc

        print(f"[SAM3] Loading model from {checkpoint_path} on {device}...")

        # Build model with both segmentation and interactivity enabled
        model = sam3.build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        print("[SAM3] Model loaded.")

        # Create processor for text-based segmentation
        processor = Sam3Processor(model, device=device)

        # Get interactive predictor for point/box prompts
        predictor = model.inst_interactive_predictor

        return processor, predictor

    def _ensure_mask(self, mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(image_shape, dtype=bool)
        if mask.dtype != np.bool_:
            return (mask > 0.5).astype(bool)
        return mask

    def _normalize_mask_to_image(self, mask: np.ndarray, image_shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Normalize SAM3 outputs to a single boolean mask with shape (H, W)."""
        image_h, image_w = image_shape
        arr = np.asarray(mask)

        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[2] <= 4:
                arr = np.max(arr, axis=2)
            else:
                arr = np.max(arr, axis=0)

        if arr.ndim != 2:
            return None

        if arr.shape == (image_w, image_h):
            arr = arr.T
        if arr.shape != (image_h, image_w):
            arr = cv2.resize(arr.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        return arr.astype(bool) if arr.dtype == np.bool_ else arr > 0.5

    def segment(self, image_bgr: np.ndarray, prompts: Iterable[str]) -> SegmentationBatch:
        """Segment using text prompts via Sam3Processor."""
        rgb_image = image_bgr[:, :, ::-1].copy()  # BGR to RGB
        image_shape = image_bgr.shape[:2]
        #pil_image = Image.fromarray(rgb_image)  # SAM3 processor expects PIL Image

        # Initialize state with image
        state = self.processor.set_image(rgb_image)

        all_masks = []
        all_boxes = []
        all_scores = []
        all_phrases = []

        # Process each text prompt
        for prompt in prompts:
            try:
                state = self.processor.set_text_prompt(prompt, state)
                print(f"[SAM3] State keys after prompt '{prompt}': {list(state.keys())}")

                if "masks" in state and state["masks"] is not None:
                    # Convert tensors to float32 before numpy (BFloat16 not supported by numpy)
                    masks_tensor = state["masks"].float()
                    masks = masks_tensor.cpu().numpy()
                    boxes = state["boxes"].float().cpu().numpy() if "boxes" in state else None
                    scores = state["scores"].float().cpu().numpy() if "scores" in state else None

                    # Debug: log mask shapes
                    print(f"[SAM3] Raw masks tensor shape: {masks_tensor.shape}, numpy shape: {masks.shape}")

                    for i in range(len(masks)):
                        raw_mask = masks[i]
                        mask = self._normalize_mask_to_image(raw_mask, image_shape)
                        if mask is None:
                            print(
                                f"[SAM3] Warning: skipping invalid mask shape {np.asarray(raw_mask).shape} "
                                f"for image shape {image_shape} (prompt='{prompt}')"
                            )
                            continue

                        all_masks.append(mask)
                        if boxes is not None and len(boxes) > i:
                            all_boxes.append(boxes[i])
                        else:
                            # Derive box from mask
                            ys, xs = np.where(mask)
                            if len(xs) > 0 and len(ys) > 0:
                                all_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                            else:
                                all_boxes.append([0.0, 0.0, 0.0, 0.0])
                        if scores is not None and len(scores) > i:
                            all_scores.append(float(scores[i]))
                        else:
                            all_scores.append(1.0)
                        all_phrases.append(prompt)

                # Reset prompts for next iteration
                self.processor.reset_all_prompts(state)

            except Exception as e:
                print(f"[SAM3] Warning: Failed to process prompt '{prompt}': {e}")
                continue

        if not all_masks:
            return SegmentationBatch(detections=sv.Detections.empty(), metadata={"prompts": list(prompts)})

        masks_arr = np.array(all_masks, dtype=bool)
        boxes_arr = np.array(all_boxes, dtype=float)
        scores_arr = np.array(all_scores, dtype=float)
        class_ids = np.zeros(len(all_masks), dtype=int)

        detections = sv.Detections(xyxy=boxes_arr, mask=masks_arr, confidence=scores_arr, class_id=class_ids)
        return SegmentationBatch(detections=detections, metadata={"prompts": all_phrases})

    def segment_with_points(self, image_bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Segment using point prompts via interactive predictor."""
        if self.predictor is None:
            raise RuntimeError("Interactive predictor not available. Build model with enable_inst_interactivity=True")

        rgb_image = image_bgr[:, :, ::-1].copy()  # BGR to RGB (copy to avoid negative strides)
        self.predictor.set_image(rgb_image)
        masks, _, _ = self.predictor.predict(
            point_coords=np.asarray(points_xy),
            point_labels=np.ones(len(points_xy), dtype=np.int32),
            multimask_output=False,
        )
        return self._ensure_mask(masks[0] if masks is not None and len(masks) else None, image_bgr.shape[:2])

    def segment_with_box(self, image_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Segment using box prompt via interactive predictor."""
        if self.predictor is None:
            raise RuntimeError("Interactive predictor not available. Build model with enable_inst_interactivity=True")

        rgb_image = image_bgr[:, :, ::-1].copy()  # BGR to RGB (copy to avoid negative strides)
        self.predictor.set_image(rgb_image)
        masks, _, _ = self.predictor.predict(box=np.asarray(box_xyxy), multimask_output=False)
        return self._ensure_mask(masks[0] if masks is not None and len(masks) else None, image_bgr.shape[:2])
