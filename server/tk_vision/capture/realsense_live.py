from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .source import Frame, FrameSource


class RealSenseLiveSource(FrameSource):
    """Live Intel RealSense color stream. Optionally records the raw stream to .bag."""

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        record_to: str | Path | None = None,
    ) -> None:
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 not available. Install it or use a different FrameSource."
            ) from e

        self._rs = __import__("pyrealsense2")
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.record_to = Path(record_to) if record_to else None
        self._pipeline = None
        self._profile = None

    def __enter__(self) -> "RealSenseLiveSource":
        rs = self._rs
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, int(self.fps))
        if self.record_to is not None:
            self.record_to.parent.mkdir(parents=True, exist_ok=True)
            cfg.enable_record_to_file(str(self.record_to))
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(cfg)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def frames(self) -> Iterator[Frame]:
        if self._pipeline is None:
            raise RuntimeError("Source must be used as a context manager.")
        idx = 0
        while True:
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            arr = np.asanyarray(color.get_data())
            yield Frame(index=idx, image_bgr=arr, timestamp_s=color.get_timestamp() / 1000.0)
            idx += 1


def realsense_available() -> bool:
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        return len(list(ctx.query_devices())) > 0
    except Exception:
        return False
