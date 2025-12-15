import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, TextIO, Union

import cv2
import numpy as np


@dataclass
class SAM3Mask:
    mask: np.ndarray
    label: str
    score: Optional[float] = None


class SAM3SegmentationCollector:
    """Lightweight SAM3-based segmentation collector with tracking support."""

    IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")
    FILENAME_TEMPLATE = "sam3_{idx:05d}"

    class _Segmenter(Protocol):
        def segment(self, frame: np.ndarray, prompts: Optional[Sequence[str]] = None) -> Iterable["SAM3Mask"]:
            ...

    class _Tracker(Protocol):
        def track(self, previous_masks: Sequence["SAM3Mask"], frame: np.ndarray) -> Iterable["SAM3Mask"]:
            ...

    def __init__(
        self,
        *,
        output_dir: str,
        segmenter: Optional[_Segmenter] = None,
        tracker: Optional[_Tracker] = None,
        ontology_path: Optional[str] = None,
        label_map: Optional[MutableMapping[str, int]] = None,
    ) -> None:
        if not output_dir:
            raise ValueError("output_dir is required")
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.segmenter = segmenter
        self.tracker = tracker
        self.label_map: MutableMapping[str, int] = label_map or self._load_label_map(ontology_path)

    def _load_label_map(self, ontology_path: Optional[str]) -> MutableMapping[str, int]:
        if ontology_path and os.path.exists(ontology_path):
            with open(ontology_path, "r") as f:
                ontology = json.load(f)
            # ontology maps prompt->label; ids follow insertion order of labels
            labels = list(ontology.values())
            return {label: idx for idx, label in enumerate(labels)}
        return {}

    def _ensure_segmenter(self) -> _Segmenter:
        if self.segmenter is not None:
            return self.segmenter
        try:
            from sam3 import LanguageGroundedSAM3  # type: ignore
        except ImportError as exc:  # pragma: no cover - real SAM3 not available in tests
            raise RuntimeError("SAM3 is not installed. Provide a segmenter or install SAM3.") from exc
        segmenter = LanguageGroundedSAM3()
        if not hasattr(segmenter, "segment"):
            raise RuntimeError("Loaded SAM3 implementation does not expose a segment(frame, prompts) method.")
        self.segmenter = segmenter
        return self.segmenter

    def _iter_masks(self, detections: Iterable[Union[SAM3Mask, Mapping]]) -> List[SAM3Mask]:
        normalized: List[SAM3Mask] = []
        for det in detections or []:
            if isinstance(det, SAM3Mask):
                normalized.append(det)
            elif isinstance(det, Mapping):
                mask = det.get("mask")
                label = det.get("label")
                score = det.get("score")
                if mask is None or label is None:
                    continue
                normalized.append(SAM3Mask(mask=np.asarray(mask), label=str(label), score=score))
        return normalized

    def _assign_class_id(self, label: str) -> int:
        if label not in self.label_map:
            self.label_map[label] = len(self.label_map)
        return self.label_map[label]

    def _write_label_file(self, mask: np.ndarray, class_id: int, h: int, w: int, fh: TextIO) -> bool:
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            return False
        segment = contour.flatten().tolist()
        normalized = [val / w if idx % 2 == 0 else val / h for idx, val in enumerate(segment)]
        fh.write(f"{class_id} " + " ".join(map(str, normalized)) + "\n")
        return True

    def _save_frame(self, frame: np.ndarray, detections: Sequence[SAM3Mask], idx: int) -> bool:
        filename_base = self.FILENAME_TEMPLATE.format(idx=idx)
        image_path = os.path.join(self.images_dir, f"{filename_base}.jpg")
        label_path = os.path.join(self.labels_dir, f"{filename_base}.txt")
        saved = False
        if not detections:
            return False

        h, w = frame.shape[:2]
        with open(label_path, "w") as fh:
            for det in detections:
                class_id = self._assign_class_id(det.label)
                if self._write_label_file(det.mask, class_id, h, w, fh):
                    saved = True

        if not saved:
            if os.path.exists(label_path):
                os.remove(label_path)
            return False

        cv2.imwrite(image_path, frame)
        return True

    def collect(self, frames: Sequence[np.ndarray], prompts: Optional[Sequence[str]] = None) -> int:
        """Run SAM3 segmentation + tracking over frames and save YOLO-seg format data."""
        if not frames:
            return 0
        segmenter = self._ensure_segmenter()
        previous_masks: Optional[List[SAM3Mask]] = None
        saved = 0

        for idx, frame in enumerate(frames):
            if idx == 0 or previous_masks is None or self.tracker is None:
                detections = segmenter.segment(frame, prompts)
            else:
                detections = self.tracker.track(previous_masks, frame)
                if not detections:
                    detections = segmenter.segment(frame, prompts)

            masks = self._iter_masks(detections)
            if self._save_frame(frame, masks, idx):
                saved += 1
            previous_masks = masks

        return saved

    def load_frames_from_dir(self, images_dir: str) -> List[np.ndarray]:
        files: List[str] = []
        for pat in self.IMAGE_PATTERNS:
            files.extend(glob.glob(os.path.join(images_dir, pat)))
        frames = []
        for path in sorted(files):
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)
        return frames


__all__ = ["SAM3SegmentationCollector", "SAM3Mask"]
