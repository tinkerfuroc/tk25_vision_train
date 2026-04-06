import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from thefuzz import process

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.crop_augment import CropAugmentConfig, build_crop_variants
from yolo_tuning.vision_tuning.data_collection.input_sources import iter_frames
from yolo_tuning.vision_tuning.data_collection.sam3_backend import OfficialSAM3Backend
from yolo_tuning.vision_tuning.data_collection.tkinter_gui import DualWindowSegCollector, FrameAnnotation, FrameReviewGUI


def _has_display() -> bool:
    """Check if a GUI display is available."""
    return os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None


@dataclass
class SegEngineOptions:
    input_mode: str = "realsense"
    source_path: Optional[str] = None
    enable_crop_augment: bool = False
    crop_variants: int = 2
    crop_scale_min: float = 1.05
    crop_scale_max: float = 1.30
    max_frames: Optional[int] = None
    fuzzy_threshold: int = 80
    enable_review: bool = True  # Enable GUI review before saving
    enable_live_preview: bool = True  # Enable real-time visualization during collection


class SegmentationCollectionEngine:
    def __init__(self, config: VisionConfig, output_dir: str, options: SegEngineOptions):
        self.config = config
        self.output_dir = output_dir
        self.options = options

        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.ontology = self._load_ontology(config.ontology_path)
        self.prompts: List[str] = list(self.ontology.keys())
        self.class_names: List[str] = list(self.ontology.values())

        self.backend = OfficialSAM3Backend(config.sam3_checkpoint_path, config.device)
        self.crop_cfg = CropAugmentConfig(
            enabled=options.enable_crop_augment,
            variants=options.crop_variants,
            scale_min=options.crop_scale_min,
            scale_max=options.crop_scale_max,
        )

    def _load_ontology(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ontology file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _next_index(self) -> int:
        return len(os.listdir(self.images_dir))

    def _class_from_phrase(self, phrase: str) -> int:
        if not self.prompts:
            return 0
        match = process.extractOne(phrase, self.prompts)
        if not match:
            return 0
        matched_prompt, score = match
        if score < self.options.fuzzy_threshold:
            return 0
        class_name = self.ontology[matched_prompt]
        return self.class_names.index(class_name)

    def _mask_to_polygon(self, mask: np.ndarray, img_w: int, img_h: int) -> Optional[List[float]]:
        """Convert a binary mask to normalized polygon coordinates for YOLO segmentation."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Take the largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return None
        # Flatten and normalize
        polygon = []
        for point in contour:
            x, y = point[0]
            polygon.extend([x / img_w, y / img_h])
        return polygon

    def _save_one(self, image: np.ndarray, boxes_xyxy: np.ndarray, class_ids: np.ndarray, masks: Optional[np.ndarray] = None) -> None:
        idx = self._next_index()
        image_path = os.path.join(self.images_dir, f"{idx:06d}.jpg")
        label_path = os.path.join(self.labels_dir, f"{idx:06d}.txt")
        cv2.imwrite(image_path, image)
        h, w = image.shape[:2]
        with open(label_path, "w") as f:
            for i, (box, class_id) in enumerate(zip(boxes_xyxy, class_ids)):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue

                # If we have segmentation masks, save as polygon format
                if masks is not None and i < len(masks):
                    polygon = self._mask_to_polygon(masks[i], w, h)
                    if polygon:
                        f.write(f"{int(class_id)} " + " ".join(f"{p:.6f}" for p in polygon) + "\n")
                        continue

                # Fallback to bounding box format
                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{int(class_id)} {x_center} {y_center} {bw} {bh}\n")

    def _build_labels(self, detections, metadata: dict) -> np.ndarray:
        # If backend doesn't provide phrases, keep existing class IDs (default zero).
        phrases = metadata.get("phrases")
        if not phrases:
            return detections.class_id.astype(int)

        class_ids = []
        for phrase in phrases:
            class_ids.append(self._class_from_phrase(str(phrase)))
        return np.array(class_ids, dtype=int)

    def _review_and_edit(self, frame_annotations: List[Tuple[np.ndarray, sv.Detections, np.ndarray]]) -> Optional[List[Tuple[np.ndarray, sv.Detections, np.ndarray]]]:
        """Tkinter GUI for reviewing collected frames before saving."""
        if not frame_annotations:
            print("[Review] No frames to review.")
            return []

        # Convert to FrameAnnotation objects
        annotations = []
        for i, (image, detections, class_ids) in enumerate(frame_annotations):
            # Validate mask dimensions
            if detections.mask is not None and len(detections.mask) > 0:
                img_h, img_w = image.shape[:2]
                mask_h, mask_w = detections.mask.shape[1], detections.mask.shape[2]
                if mask_h != img_h or mask_w != img_w:
                    print(f"[Review] Skipping frame {i}: mask shape mismatch")
                    continue

            annotations.append(FrameAnnotation(
                image=image,
                xyxy=detections.xyxy.copy(),
                mask=detections.mask.copy() if detections.mask is not None else None,
                confidence=detections.confidence.copy() if detections.confidence is not None else None,
                class_id=class_ids.copy(),
            ))

        if not annotations:
            print("[Review] No valid frames to review.")
            return []

        print("\n[Review] === Frame Review Mode ===")
        print("[Review] ←/→ or j/k: Navigate | 'd': Delete frame | 's': Save all | 'q': Cancel")

        gui = FrameReviewGUI(self.class_names)
        kept_indices = gui.run(annotations)

        if kept_indices is None:
            return None
        return [frame_annotations[i] for i in kept_indices]

    def run(self) -> int:
        print(f"[SegEngine] Starting collection with prompts: {self.prompts}")
        print(f"[SegEngine] Output directory: {self.output_dir}")

        enable_gui = self.options.enable_live_preview and _has_display()
        if self.options.enable_live_preview and not enable_gui:
            print("[SegEngine] No display detected. Running in headless mode.")

        saved_count = 0
        frame_annotations: List[Tuple[np.ndarray, sv.Detections, np.ndarray]] = []

        if enable_gui:
            # Interactive mode with dual-window Tkinter GUI
            print("\n[SegEngine] === Dual-Window Collection Mode ===")
            print("[SegEngine] Live Preview: real-time camera feed")
            print("[SegEngine] Review Window: detections with Save/Skip controls")
            print("[SegEngine] Controls: 's' Save | Space Skip | 'q' Quit")

            def on_save(frame: np.ndarray, detections: sv.Detections, class_ids: np.ndarray):
                nonlocal saved_count
                self._save_one(frame, detections.xyxy, class_ids, masks=detections.mask)
                saved_count += 1

                if self.crop_cfg.enabled and detections.mask is not None:
                    crop_variants = build_crop_variants(frame, detections.mask, self.crop_cfg)
                    for crop in crop_variants:
                        self._save_one(crop, np.array([[0, 0, crop.shape[1] - 1, crop.shape[0] - 1]], dtype=float), np.array([0]))
                        saved_count += 1

            gui = DualWindowSegCollector(
                class_names=self.class_names,
                on_save=on_save,
            )
            gui.start()

            frame_iter = iter_frames(self.options.input_mode, self.options.source_path)

            for frame_idx, frame in enumerate(frame_iter):
                if not gui.is_alive():
                    break
                if self.options.max_frames is not None and frame_idx >= self.options.max_frames:
                    break

                # Run segmentation processing (live preview is handled by GUI internally)
                batch = self.backend.segment(frame, self.prompts)
                detections = batch.detections

                if frame_idx < 3:
                    mask_shape = detections.mask.shape if detections.mask is not None else None
                    print(f"[SegEngine] Frame {frame_idx}: image={frame.shape}, mask={mask_shape}")

                class_ids = self._build_labels(detections, batch.metadata)

                # Update review window with detection results
                if not gui.update_review(frame, detections, class_ids):
                    break

                if frame_idx % 30 == 0:
                    print(f"[SegEngine] Processing frame {frame_idx}, saved so far: {saved_count}")

            gui.stop()
            print(f"[SegEngine] Collection complete. Total saved: {saved_count}")
            return saved_count

        # Headless mode (no GUI)
        frame_iter = iter_frames(self.options.input_mode, self.options.source_path)

        for frame_idx, frame in enumerate(frame_iter):
            if self.options.max_frames is not None and frame_idx >= self.options.max_frames:
                break

            if frame_idx % 30 == 0:
                print(f"[SegEngine] Processing frame {frame_idx}, collected: {len(frame_annotations)}")

            batch = self.backend.segment(frame, self.prompts)
            detections = batch.detections

            if len(detections) == 0:
                continue

            class_ids = self._build_labels(detections, batch.metadata)
            frame_annotations.append((frame, detections, class_ids))

        print(f"[SegEngine] Collection phase complete. Collected {len(frame_annotations)} frame(s).")

        # Review phase (if enabled)
        if self.options.enable_review and frame_annotations:
            print("[SegEngine] Launching review GUI...")
            reviewed = self._review_and_edit(frame_annotations)
            if reviewed is None:
                print("[SegEngine] Review cancelled. No frames saved.")
                return 0
            frame_annotations = reviewed

        # Save frames
        for frame, detections, class_ids in frame_annotations:
            self._save_one(frame, detections.xyxy, class_ids, masks=detections.mask)
            saved_count += 1

            if self.crop_cfg.enabled and detections.mask is not None:
                crop_variants = build_crop_variants(frame, detections.mask, self.crop_cfg)
                for crop in crop_variants:
                    self._save_one(crop, np.array([[0, 0, crop.shape[1] - 1, crop.shape[0] - 1]], dtype=float), np.array([0]))
                    saved_count += 1

        print(f"[SegEngine] Collection complete. Total saved: {saved_count}")
        return saved_count
