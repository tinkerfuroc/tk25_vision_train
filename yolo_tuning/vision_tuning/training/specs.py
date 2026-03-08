from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainJob:
    task: Literal["det", "seg"]
    project_name: str
    default_epochs: int
    default_batch: int
    default_imgsz: int = 640
