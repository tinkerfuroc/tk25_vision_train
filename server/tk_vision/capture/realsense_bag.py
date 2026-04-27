from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .source import Frame, FrameSource


class RealSenseBagSource(FrameSource):
    """Replay a previously recorded RealSense .bag (color stream only)."""

    def __init__(self, bag_path: str | Path) -> None:
        try:
            import pyrealsense2  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pyrealsense2 not available; cannot replay .bag.") from e

        self._rs = __import__("pyrealsense2")
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(self.bag_path)
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._pipeline = None

    def __enter__(self) -> "RealSenseBagSource":
        rs = self._rs
        cfg = rs.config()
        cfg.enable_device_from_file(str(self.bag_path), repeat_playback=False)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)
        # Disable real-time so we can decode as fast as we can.
        device = profile.get_device()
        try:
            playback = device.as_playback()
            playback.set_real_time(False)
        except Exception:
            pass
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = color_stream.width()
        self.height = color_stream.height()
        self.fps = float(color_stream.fps())
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
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                # End of bag.
                return
            color = frames.get_color_frame()
            if not color:
                continue
            arr = np.asanyarray(color.get_data())
            yield Frame(index=idx, image_bgr=arr, timestamp_s=color.get_timestamp() / 1000.0)
            idx += 1
