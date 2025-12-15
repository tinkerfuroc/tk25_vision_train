"""Convenient wrappers around live test scripts."""

from typing import Optional

from yolo_tuning.test_new_model import run_live_test as _run_bbox_live
from yolo_tuning.test_new_model_seg import run_live_test as _run_seg_live


def run_live_detection(model_path: Optional[str] = None) -> None:
    """Run live detection test (bbox)."""
    _run_bbox_live(model_path or "yolo_finetuned_best.pt")


def run_live_segmentation(model_path: Optional[str] = None) -> None:
    """Run live segmentation test."""
    _run_seg_live(model_path or "yolo_seg_finetuned_best.pt")


__all__ = ["run_live_detection", "run_live_segmentation"]
