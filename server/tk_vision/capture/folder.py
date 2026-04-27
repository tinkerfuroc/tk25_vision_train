from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from .source import Frame, FrameSource


class FolderSource(FrameSource):
    """Replay an ordered directory of .jpg/.png frames. Useful for headless tests."""

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, folder: str | Path, fps: float = 30.0) -> None:
        self._folder = Path(folder)
        if not self._folder.is_dir():
            raise FileNotFoundError(f"Folder source: {self._folder} is not a directory")
        self._files = sorted(
            p for p in self._folder.iterdir() if p.suffix.lower() in self.EXTENSIONS
        )
        if not self._files:
            raise FileNotFoundError(f"Folder source: no images in {self._folder}")
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise RuntimeError(f"Failed to decode {self._files[0]}")
        self.height, self.width = first.shape[:2]
        self.fps = fps

    @property
    def frame_count(self) -> int:
        return len(self._files)

    def __enter__(self) -> "FolderSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def frames(self) -> Iterator[Frame]:
        for i, path in enumerate(self._files):
            img = cv2.imread(str(path))
            if img is None:
                continue
            yield Frame(index=i, image_bgr=img, timestamp_s=i / self.fps)
