import os
import threading
from typing import Generator, Optional

import cv2
import numpy as np


def iter_image_folder_frames(source_path: str) -> Generator[np.ndarray, None, None]:
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Image folder not found: {source_path}")

    names = sorted(os.listdir(source_path))
    image_names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if not image_names:
        raise FileNotFoundError(f"No images found in folder: {source_path}")

    for name in image_names:
        image = cv2.imread(os.path.join(source_path, name))
        if image is None:
            continue
        yield image


def iter_video_frames(source_path: str) -> Generator[np.ndarray, None, None]:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video file not found: {source_path}")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class RealSenseCamera:
    """RealSense camera manager with shared frame buffer for GUI and processing."""

    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline: Optional[rs.pipeline] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.poll_thread: Optional[threading.Thread] = None
        self.frame_count = 0

    def start(self) -> None:
        """Start the camera and background polling thread."""
        if self.pipeline is not None:
            # Already started
            return
        print("[RealSense] Initializing camera...")
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.bgr8, 30)
        self.pipeline.start(config)
        print("[RealSense] Camera started.")

        self.stop_event.clear()
        self.poll_thread = threading.Thread(target=self._poll_frames, daemon=True)
        self.poll_thread.start()

    def _poll_frames(self) -> None:
        """Background thread that continuously polls frames."""
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color_frame = frames.get_color_frame()
                if color_frame:
                    with self.frame_lock:
                        self.latest_frame = np.asanyarray(color_frame.get_data())
                        self.frame_count += 1
            except Exception:
                pass

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (non-blocking)."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_frame_count(self) -> int:
        """Get the total frame count."""
        with self.frame_lock:
            return self.frame_count

    def stop(self) -> None:
        """Stop the camera."""
        print("[RealSense] Stopping camera...")
        self.stop_event.set()
        if self.poll_thread:
            self.poll_thread.join(timeout=1.0)
        if self.pipeline:
            self.pipeline.stop()
        print("[RealSense] Camera stopped.")


# Global camera instance for shared access
_camera_instance: Optional[RealSenseCamera] = None


def get_shared_camera() -> RealSenseCamera:
    """Get or create the shared camera instance."""
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = RealSenseCamera()
    return _camera_instance


def iter_realsense_frames() -> Generator[np.ndarray, None, None]:
    """Yield frames from RealSense camera with background polling.

    Uses a background thread to continuously poll frames, preventing buffer
    overflow and timeout errors during slow processing (e.g., segmentation).
    """
    camera = get_shared_camera()
    camera.start()

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                yield frame
            else:
                # Wait for first frame
                import time
                time.sleep(0.01)
    finally:
        camera.stop()


def iter_frames(input_mode: str, source_path: Optional[str]) -> Generator[np.ndarray, None, None]:
    if input_mode == "realsense":
        yield from iter_realsense_frames()
        return

    if input_mode == "images":
        if not source_path:
            raise ValueError("--source-path is required when --input-mode=images")
        yield from iter_image_folder_frames(source_path)
        return

    if input_mode == "video":
        if not source_path:
            raise ValueError("--source-path is required when --input-mode=video")
        yield from iter_video_frames(source_path)
        return

    raise ValueError(f"Unsupported input mode: {input_mode}")
