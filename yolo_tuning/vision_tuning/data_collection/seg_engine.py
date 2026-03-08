import json
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from thefuzz import process

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.crop_augment import CropAugmentConfig, build_crop_variants
from yolo_tuning.vision_tuning.data_collection.input_sources import iter_frames
from yolo_tuning.vision_tuning.data_collection.sam3_backend import OfficialSAM3Backend


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

    def _save_one(self, image: np.ndarray, boxes_xyxy: np.ndarray, class_ids: np.ndarray) -> None:
        idx = self._next_index()
        image_path = os.path.join(self.images_dir, f"{idx:06d}.jpg")
        label_path = os.path.join(self.labels_dir, f"{idx:06d}.txt")
        cv2.imwrite(image_path, image)
        h, w = image.shape[:2]
        with open(label_path, "w") as f:
            for box, class_id in zip(boxes_xyxy, class_ids):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
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

    def run(self) -> int:
        saved_count = 0
        frame_iter = iter_frames(self.options.input_mode, self.options.source_path)
        for frame_idx, frame in enumerate(frame_iter):
            if self.options.max_frames is not None and frame_idx >= self.options.max_frames:
                break

            batch = self.backend.segment(frame, self.prompts)
            detections = batch.detections
            if len(detections) == 0:
                continue

            class_ids = self._build_labels(detections, batch.metadata)
            self._save_one(frame, detections.xyxy, class_ids)
            saved_count += 1

            if self.crop_cfg.enabled and detections.mask is not None:
                crop_variants = build_crop_variants(frame, detections.mask, self.crop_cfg)
                for crop in crop_variants:
                    self._save_one(crop, np.array([[0, 0, crop.shape[1] - 1, crop.shape[0] - 1]], dtype=float), np.array([0]))
                    saved_count += 1

        return saved_count
