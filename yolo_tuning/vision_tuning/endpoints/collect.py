from typing import Optional

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.collectors import (
    launch_bbox_collection,
    launch_seg_collection,
    launch_seg_stream_collection,
)
from yolo_tuning.vision_tuning.data_collection.splitter import split_yolo_dataset


def collect_bbox(config: VisionConfig, *, dataset_dir: Optional[str] = None) -> None:
    launch_bbox_collection(config, output_dir=dataset_dir)


def collect_seg(
    config: VisionConfig,
    *,
    dataset_dir: Optional[str] = None,
    input_mode: str = "realsense",
    source_path: Optional[str] = None,
    enable_crop_augment: bool = False,
    crop_variants: int = 2,
    crop_scale_min: float = 1.05,
    crop_scale_max: float = 1.30,
    max_frames: Optional[int] = None,
) -> None:
    launch_seg_collection(
        config,
        output_dir=dataset_dir,
        input_mode=input_mode,
        source_path=source_path,
        enable_crop_augment=enable_crop_augment,
        crop_variants=crop_variants,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        max_frames=max_frames,
    )


def collect_seg_stream(
    config: VisionConfig,
    *,
    dataset_dir: Optional[str] = None,
    input_mode: str = "realsense",
    source_path: Optional[str] = None,
    enable_crop_augment: bool = False,
    crop_variants: int = 2,
    crop_scale_min: float = 1.05,
    crop_scale_max: float = 1.30,
    max_frames: Optional[int] = 300,
) -> None:
    launch_seg_stream_collection(
        config,
        output_dir=dataset_dir,
        input_mode=input_mode,
        source_path=source_path,
        enable_crop_augment=enable_crop_augment,
        crop_variants=crop_variants,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        max_frames=max_frames,
    )


def split_dataset(
    config: VisionConfig,
    *,
    dataset_dir: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    root = dataset_dir or config.dataset_dir
    return split_yolo_dataset(root, train_ratio=train_ratio, seed=seed or config.seed)
