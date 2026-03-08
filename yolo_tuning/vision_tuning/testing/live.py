"""Backward-compatible live testing wrappers."""

from typing import Optional

from yolo_tuning.vision_tuning.testing.workflow import run_detection_live, run_segmentation_live


def run_live_detection(model_path: Optional[str] = None) -> None:
    run_detection_live(model_path=model_path)


def run_live_segmentation(model_path: Optional[str] = None) -> None:
    run_segmentation_live(model_path=model_path)


__all__ = ["run_live_detection", "run_live_segmentation"]
