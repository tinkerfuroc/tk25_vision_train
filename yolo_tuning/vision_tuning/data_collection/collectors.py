"""Shim kept for backward compatibility; prefer capture_* modules."""

from typing import Optional

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.capture_bbox import run_bbox_collection
from yolo_tuning.vision_tuning.data_collection.capture_seg import run_seg_collection
from yolo_tuning.vision_tuning.data_collection.capture_seg_stream import run_seg_stream_collection


def launch_bbox_collection(config: VisionConfig, output_dir: Optional[str] = None) -> None:
    run_bbox_collection(config, output_dir)


def launch_seg_collection(config: VisionConfig, output_dir: Optional[str] = None) -> None:
    run_seg_collection(config, output_dir)


def launch_seg_stream_collection(config: VisionConfig, output_dir: Optional[str] = None) -> None:
    run_seg_stream_collection(config, output_dir)
