"""Shim kept for backward compatibility; prefer capture_* modules."""

from typing import Optional

from yolo_tuning.vision_tuning.config import VisionConfig


def launch_bbox_collection(config: VisionConfig, output_dir: Optional[str] = None) -> None:
    from yolo_tuning.vision_tuning.data_collection.capture_bbox import run_bbox_collection

    run_bbox_collection(config, output_dir)


def launch_seg_collection(
    config: VisionConfig,
    output_dir: Optional[str] = None,
    *,
    input_mode: str = "realsense",
    source_path: Optional[str] = None,
    enable_crop_augment: bool = False,
    crop_variants: int = 2,
    crop_scale_min: float = 1.05,
    crop_scale_max: float = 1.30,
    max_frames: Optional[int] = None,
    enable_review: bool = True,
    enable_live_preview: bool = True,
) -> None:
    from yolo_tuning.vision_tuning.data_collection.capture_seg import run_seg_collection

    run_seg_collection(
        config,
        output_dir,
        input_mode=input_mode,
        source_path=source_path,
        enable_crop_augment=enable_crop_augment,
        crop_variants=crop_variants,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        max_frames=max_frames,
        enable_review=enable_review,
        enable_live_preview=enable_live_preview,
    )


def launch_seg_stream_collection(
    config: VisionConfig,
    output_dir: Optional[str] = None,
    *,
    input_mode: str = "realsense",
    source_path: Optional[str] = None,
    enable_crop_augment: bool = False,
    crop_variants: int = 2,
    crop_scale_min: float = 1.05,
    crop_scale_max: float = 1.30,
    max_frames: Optional[int] = 300,
    enable_review: bool = True,
    enable_live_preview: bool = True,
) -> None:
    from yolo_tuning.vision_tuning.data_collection.capture_seg_stream import run_seg_stream_collection

    run_seg_stream_collection(
        config,
        output_dir,
        input_mode=input_mode,
        source_path=source_path,
        enable_crop_augment=enable_crop_augment,
        crop_variants=crop_variants,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        max_frames=max_frames,
        enable_review=enable_review,
        enable_live_preview=enable_live_preview,
    )
