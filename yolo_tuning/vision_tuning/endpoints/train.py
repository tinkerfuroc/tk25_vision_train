from typing import Optional, Tuple

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.training import train_detector, train_segmenter


def train_bbox(
    config: VisionConfig,
    *,
    dataset_dir: Optional[str] = None,
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 640,
    train_ratio: float = 0.8,
) -> Tuple[str, str]:
    return train_detector(
        config,
        dataset_path=dataset_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        train_ratio=train_ratio,
    )


def train_seg(
    config: VisionConfig,
    *,
    dataset_dir: Optional[str] = None,
    epochs: int = 250,
    batch: int = 4,
    imgsz: int = 640,
    train_ratio: float = 0.8,
) -> Tuple[str, str]:
    return train_segmenter(
        config,
        dataset_path=dataset_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        train_ratio=train_ratio,
    )
