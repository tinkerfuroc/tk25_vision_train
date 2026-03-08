from typing import Optional

from yolo_tuning.vision_tuning.testing import run_live_detection, run_live_segmentation


def test_bbox(model_path: Optional[str] = None) -> None:
    run_live_detection(model_path)


def test_seg(model_path: Optional[str] = None) -> None:
    run_live_segmentation(model_path)
