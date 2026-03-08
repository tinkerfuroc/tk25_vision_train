from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LiveTestJob:
    task: Literal["bbox", "seg"]
    default_model: str
    width: int
    height: int
    fps: int


DETECTION_LIVE_JOB = LiveTestJob(task="bbox", default_model="yolo_finetuned_best.pt", width=640, height=480, fps=30)
SEGMENTATION_LIVE_JOB = LiveTestJob(
    task="seg",
    default_model="yolo_seg_finetuned_best.pt",
    width=1280,
    height=720,
    fps=15,
)
