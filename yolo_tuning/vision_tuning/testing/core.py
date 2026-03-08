from typing import Literal

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


def run_realsense_live_test(
    model_path: str,
    *,
    task: Literal["bbox", "seg"] = "bbox",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> None:
    import pyrealsense2 as rs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)

    if task == "seg":
        annotator = sv.MaskAnnotator()
    else:
        annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            result = model(image, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated = annotator.annotate(scene=image.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            cv2.imshow("Live Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
