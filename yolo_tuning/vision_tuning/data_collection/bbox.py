"""RealSense bounding-box dataset collection."""

import copy
import json
import os
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
import torch
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from dotenv import load_dotenv

# Ensure env vars are available for defaults
load_dotenv()


def pre_cache_tokenizer() -> None:
    """Downloads and caches the tokenizer model from Hugging Face."""
    print("Pre-caching tokenizer model 'bert-base-uncased'...")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("bert-base-uncased")
        print("Tokenizer is cached.")
    except Exception as e:  # pragma: no cover - network/hardware dependent
        print(f"Failed to download tokenizer: {e}")
        print("Please check your internet connection and firewall settings.")
        print("You may need to configure HTTP/HTTPS proxies if you are behind a firewall.")


class RealSenseBBoxCollector:
    """Live RealSense capture with GroundingDINO-assisted box labeling."""

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        ontology_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        print("Initializing RealSenseBBoxCollector...")
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.ontology_path = ontology_path or os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
        self.ontology = self._load_ontology()
        if self.ontology:
            self.base_model = GroundingDINO(ontology=self.ontology)
            if self.device.type == "cuda":
                try:
                    self.base_model.dino_model.model.to(self.device)
                    self.base_model.dino_model.device = self.device
                    print("Moved GroundingDINO model to CUDA device.")
                except AttributeError:
                    print("Could not move model to CUDA. It might not be supported by this version of autodistill-groundingdino.")
        else:
            self.base_model = None

        print("Starting RealSense camera pipeline...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        print("RealSense camera pipeline started.")

        self.output_dir = output_dir or os.getenv("DATASET_DIR", "dataset")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def _load_ontology(self):
        print("Loading ontology...")
        try:
            with open(self.ontology_path, "r") as f:
                ontology_data = json.load(f)
            print(f"Loaded ontology from {self.ontology_path}")
            return CaptionOntology(ontology_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load ontology file: {e}. Aborting.")
            return None

    def save_data(self, image, predictions):
        # Save image
        img_idx = len(os.listdir(self.images_dir))
        img_filename = os.path.join(self.images_dir, f"{img_idx:06d}.jpg")
        cv2.imwrite(img_filename, image)

        # Save labels in YOLO format
        label_filename = os.path.join(self.labels_dir, f"{img_idx:06d}.txt")
        image_height, image_width = image.shape[:2]

        with open(label_filename, "w") as f:
            for box, class_id in zip(predictions.xyxy, predictions.class_id):
                x_center = (box[0] + box[2]) / 2 / image_width
                y_center = (box[1] + box[3]) / 2 / image_height
                box_width = (box[2] - box[0]) / image_width
                box_height = (box[3] - box[1]) / image_height

                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

    def run(self):
        if not self.base_model:
            return

        print("Starting dataset creation...")
        print("--- Image Controls ---")
        print(" 's': Save approved detections and go to the next image.")
        print(" 'space': Skip this image without saving.")
        print(" 'q': Quit the application.")
        print("--- Detection Controls ---")
        print(" 'down arrow': Select next detection.")
        print(" 'up arrow': Select previous detection.")
        print(" 'd': Delete the currently selected detection.")

        box_annotator = sv.BoxAnnotator(thickness=2, color=sv.ColorPalette.ROBOFLOW)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_position=sv.Position.BOTTOM_LEFT)
        highlight_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.RED)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                cv_image = np.asanyarray(color_frame.get_data())
                pad_top, pad_bottom, pad_left, pad_right = 50, 50, 50, 50
                border_color = [0, 0, 0]

                predictions = self.base_model.predict(cv_image)

                if len(predictions) == 0:
                    display_image = cv2.copyMakeBorder(
                        cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border_color
                    )
                    cv2.imshow("Image", display_image)
                    print("No detections found. Press any key to skip, or 'q' to quit.")
                    key = cv2.waitKeyEx(0)
                    if key == ord("q"):
                        print("Quitting.")
                        break
                    else:
                        print("Skipped image (no detections).")
                        continue

                display_predictions = copy.deepcopy(predictions)
                display_predictions.xyxy[:, [0, 2]] += pad_left
                display_predictions.xyxy[:, [1, 3]] += pad_top

                kept_indices = list(range(len(predictions)))
                selected_idx = 0

                while True:
                    display_image = cv2.copyMakeBorder(
                        cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border_color
                    )

                    if not kept_indices:
                        cv2.imshow("Image", display_image)
                    else:
                        detections_to_show = display_predictions[kept_indices]

                        labels = [
                            f"{self.ontology.classes()[class_id]} {confidence:0.2f}"
                            for class_id, confidence in zip(detections_to_show.class_id, detections_to_show.confidence)
                        ]

                        annotated_image = box_annotator.annotate(scene=display_image, detections=detections_to_show)
                        annotated_image = label_annotator.annotate(
                            scene=annotated_image, detections=detections_to_show, labels=labels
                        )

                        selected_detection = detections_to_show[selected_idx]
                        annotated_image = highlight_annotator.annotate(scene=annotated_image, detections=selected_detection)

                        cv2.imshow("Image", annotated_image)

                    key = cv2.waitKeyEx(0)

                    if key == ord("q"):
                        self.pipeline.stop()
                        cv2.destroyAllWindows()
                        print("Quitting application.")
                        return

                    elif key == ord("s"):
                        if kept_indices:
                            final_predictions = predictions[kept_indices]
                            self.save_data(cv_image, final_predictions)
                            print(f"Saved image with {len(final_predictions)} detections.")
                        else:
                            print("No detections to save.")
                        break

                    elif key == 32:
                        print("Skipped image.")
                        break

                    if not kept_indices:
                        continue

                    if key == 65364:
                        selected_idx = (selected_idx + 1) % len(kept_indices)

                    elif key == 65362:
                        selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)

                    elif key == ord("d"):
                        kept_indices.pop(selected_idx)
                        if not kept_indices:
                            selected_idx = 0
                        else:
                            selected_idx %= len(kept_indices)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("RealSense camera pipeline stopped.")


__all__ = ["RealSenseBBoxCollector", "pre_cache_tokenizer"]
