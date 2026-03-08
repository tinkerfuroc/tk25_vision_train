import os
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


def iter_realsense_frames() -> Generator[np.ndarray, None, None]:
    import pyrealsense2 as rs  # Lazy import to keep offline workflows hardware-agnostic.

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            yield np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


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
