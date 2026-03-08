import os
from typing import Optional

from yolo_tuning.vision_tuning.testing.core import run_realsense_live_test
from yolo_tuning.vision_tuning.testing.specs import DETECTION_LIVE_JOB, SEGMENTATION_LIVE_JOB, LiveTestJob


def _resolve_model(model_path: Optional[str], job: LiveTestJob) -> str:
    return model_path or job.default_model


def run_live_job(job: LiveTestJob, model_path: Optional[str] = None) -> None:
    resolved = _resolve_model(model_path, job)
    if not os.path.exists(resolved):
        print(f"Error: Model file not found at {resolved}")
        return
    run_realsense_live_test(
        resolved,
        task=job.task,
        width=job.width,
        height=job.height,
        fps=job.fps,
    )


def run_detection_live(model_path: Optional[str] = None) -> None:
    run_live_job(DETECTION_LIVE_JOB, model_path=model_path)


def run_segmentation_live(model_path: Optional[str] = None) -> None:
    run_live_job(SEGMENTATION_LIVE_JOB, model_path=model_path)
