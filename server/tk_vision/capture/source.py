from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(slots=True)
class Frame:
    index: int
    image_bgr: np.ndarray
    timestamp_s: float | None = None


class FrameSource(ABC):
    """Common interface for live RealSense, .bag replay, and folder replay."""

    width: int
    height: int
    fps: float

    @abstractmethod
    def __enter__(self) -> "FrameSource":
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        ...
