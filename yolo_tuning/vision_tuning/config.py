import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VisionConfig:
    """Centralized configuration for dataset paths, weights, and runtime defaults."""

    dataset_dir: str = field(default_factory=lambda: os.getenv("DATASET_DIR", "dataset"))
    seg_dataset_dir: str = field(
        default_factory=lambda: os.getenv("DATASET_SEG_DIR", os.getenv("DATASET_DIR", "dataset_seg"))
    )
    ontology_path: str = field(
        default_factory=lambda: os.getenv(
            "ONTOLOGY_PATH",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource", "ontology.json")),
        )
    )
    checkpoint_dir: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DIR", "runs"))
    device: str = field(default_factory=_default_device)
    detection_weights: str = field(default_factory=lambda: os.getenv("YOLO_BASE_WEIGHTS", "yolo11s.pt"))
    segmentation_weights: str = field(default_factory=lambda: os.getenv("YOLO_SEG_WEIGHTS", "yolo11s-seg.pt"))
    seed: int = field(default_factory=lambda: int(os.getenv("VISION_TRAIN_SEED", "42")))

    def override(
        self,
        *,
        dataset_dir: Optional[str] = None,
        seg_dataset_dir: Optional[str] = None,
        ontology_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "VisionConfig":
        """Return a copy with overrides applied."""
        return VisionConfig(
            dataset_dir=dataset_dir or self.dataset_dir,
            seg_dataset_dir=seg_dataset_dir or self.seg_dataset_dir,
            ontology_path=ontology_path or self.ontology_path,
            checkpoint_dir=checkpoint_dir or self.checkpoint_dir,
            device=device or self.device,
            detection_weights=self.detection_weights,
            segmentation_weights=self.segmentation_weights,
            seed=self.seed,
        )
