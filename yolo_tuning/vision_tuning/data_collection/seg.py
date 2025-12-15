"""RealSense segmentation-mask collection."""

import copy
import json
import os
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
import torch
from dotenv import load_dotenv
from lang_sam import LangSAM
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from thefuzz import process

# Load environment variables for defaults
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


class RealSenseSegCollector:
    """Live RealSense segmentation capture with LangSAM + SAM assist."""

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        ontology_path: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        print("Initializing RealSenseSegCollector...")
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.ontology_path = ontology_path or os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
        self.ontology = self._load_ontology()
        self.prompts = list(self.ontology.keys()) if self.ontology else []
        self.class_names = list(self.ontology.values()) if self.ontology else []
        self.prompt_to_class = self.ontology if self.ontology else {}

        print("Loading LangSAM model...")
        self.base_model = LangSAM()
        print("LangSAM model loaded.")

        print("Loading SAM model for manual annotation...")
        sam_type = "vit_b"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_sam_path = os.path.join(script_dir, "..", "..", "sam_vit_b_01ec64.pth")
        sam_checkpoint = sam_checkpoint or os.getenv("SAM_CHECKPOINT_PATH", default_sam_path)
        if not os.path.exists(sam_checkpoint):
            print(f"SAM checkpoint not found at {sam_checkpoint}. Please download it or update the path in your .env file.")
            self.sam_predictor = None
        else:
            sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM model loaded.")

        print("Starting RealSense camera pipeline...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        print("RealSense camera pipeline started.")

        self.output_dir = output_dir or os.getenv("DATASET_SEG_DIR", os.getenv("DATASET_DIR", "dataset_seg"))
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.mouse_points = []
        self.current_class_idx = 0
        self.temp_manual_mask = None

    def _load_ontology(self):
        print("Loading ontology...")
        try:
            with open(self.ontology_path, "r") as f:
                ontology_data = json.load(f)
            print(f"Loaded ontology from {self.ontology_path}")
            return ontology_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load ontology file: {e}. Aborting.")
            return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            top, left = 50, 50
            if x >= left and y >= top:
                adjusted_x = x - left
                adjusted_y = y - top
                self.mouse_points.append((adjusted_x, adjusted_y))
                print(f"Added point: ({adjusted_x}, {adjusted_y})")

    def _create_padded_masks(self, masks, image_shape, top, bottom, left, right):
        padded_masks = []
        for mask in masks:
            padded_mask = np.zeros((image_shape[0] + top + bottom, image_shape[1] + left + right), dtype=bool)
            padded_mask[top : top + image_shape[0], left : left + image_shape[1]] = mask
            padded_masks.append(padded_mask)
        return np.array(padded_masks)

    def _prepare_predictions_for_display(self, predictions, left, top, image_shape, padding):
        predictions_for_display = copy.deepcopy(predictions)
        predictions_for_display.xyxy += np.array([left, top, left, top])

        top_pad, bottom_pad, left_pad, right_pad = padding
        padded_masks = self._create_padded_masks(
            predictions_for_display.mask, image_shape, top_pad, bottom_pad, left_pad, right_pad
        )
        predictions_for_display.mask = padded_masks

        return predictions_for_display

    def _create_labels(self, predictions, prefix=""):
        labels = [
            f"{prefix}#{idx} {self.class_names[cid]} {conf:.2f}"
            for idx, (cid, conf) in enumerate(zip(predictions.class_id, predictions.confidence))
        ]
        return labels

    def _render_thumbnail(self, annotated_image, mask, class_id, top, is_preview=False):
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            return annotated_image

        y1, x1 = mask_coords.min(axis=0)
        y2, x2 = mask_coords.max(axis=0)
        crop_h, crop_w = y2 - y1, x2 - x1

        if crop_h <= 0 or crop_w <= 0:
            return annotated_image

        cropped_region = annotated_image[y1:y2, x1:x2].copy()

        max_thumb_h = top - 10
        scale = min(max_thumb_h / crop_h, 150 / crop_w, 1.0)
        thumb_w = int(crop_w * scale)
        thumb_h = int(crop_h * scale)

        if thumb_h > 0 and thumb_w > 0:
            thumbnail = cv2.resize(cropped_region, (thumb_w, thumb_h))
            thumb_x = annotated_image.shape[1] - thumb_w - 10
            thumb_y = 5
            annotated_image[thumb_y : thumb_y + thumb_h, thumb_x : thumb_x + thumb_w] = thumbnail
            color = (255, 0, 0) if is_preview else (0, 0, 255)
            cv2.rectangle(annotated_image, (thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h), color, 2)

        return annotated_image

    def _run_automatic_segmentation(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        text_prompt = ". ".join(self.prompts)
        print("Using text prompt:", text_prompt)
        predictions = self.base_model.predict([pil_image], [text_prompt])
        print(predictions[0].keys())

        all_detections = []
        if predictions:
            result = predictions[0]
            boxes = result["boxes"]
            masks = result["masks"]
            scores = result["scores"]
            labels = result["labels"]

            detections = sv.Detections(
                xyxy=np.array(boxes),
                mask=np.array(masks),
                confidence=np.array(scores),
                class_id=np.array(labels),
            )
            all_detections.append(detections)

        return all_detections[0] if all_detections else None

    def _manual_segmentation(self, cv_image):
        if self.sam_predictor is None:
            print("SAM predictor not available. Cannot perform manual segmentation.")
            return None

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(rgb_image)

        masks, _, _ = self.sam_predictor.predict(
            point_coords=np.array(self.mouse_points),
            point_labels=np.ones(len(self.mouse_points)),
            multimask_output=False,
        )

        self.mouse_points.clear()
        if masks is None or len(masks) == 0:
            print("No mask generated from manual points.")
            return None

        return masks[0]

    def save_data(self, image, detections):
        img_idx = len(os.listdir(self.images_dir))
        img_filename = os.path.join(self.images_dir, f"{img_idx:06d}.jpg")
        cv2.imwrite(img_filename, image)

        label_filename = os.path.join(self.labels_dir, f"{img_idx:06d}.txt")
        image_height, image_width = image.shape[:2]

        with open(label_filename, "w") as f:
            for mask, class_id in zip(detections.mask, detections.class_id):
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x_center = np.mean(xs) / image_width
                y_center = np.mean(ys) / image_height
                box_width = (np.max(xs) - np.min(xs)) / image_width
                box_height = (np.max(ys) - np.min(ys)) / image_height
                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

    def run(self):
        if not self.base_model:
            return

        print("Starting dataset creation...")
        print("Press 'Space' to capture current image. Use 'Up'/'Down' to select segments, 'd' to delete.")
        print("Press 'm' for manual mode: click points, press 'Enter' to generate, choose label, 'a' to add.")
        print("Press 's' to save, 'esc' to discard, 'q' to quit.")

        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT)
        padding = (50, 50, 50, 50)
        pad_top, pad_bottom, pad_left, pad_right = padding

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                cv_image = np.asanyarray(color_frame.get_data())
                display_image = cv2.copyMakeBorder(
                    cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key != ord(" "):
                    continue

                detections = self._run_automatic_segmentation(cv_image)
                if detections is None:
                    print("No detections; skipping frame.")
                    continue

                display_predictions = self._prepare_predictions_for_display(
                    detections, pad_left, pad_top, cv_image.shape, padding
                )
                kept_indices = list(range(len(display_predictions)))
                selected_idx = 0
                manual_mode = False
                temp_manual_mask = None

                while True:
                    annotated_image = mask_annotator.annotate(
                        scene=display_image.copy(), detections=display_predictions[kept_indices]
                    )
                    labels = self._create_labels(display_predictions[kept_indices], prefix="Auto")
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=display_predictions[kept_indices], labels=labels
                    )

                    if temp_manual_mask is not None:
                        annotated_image[temp_manual_mask] = (0, 0, 255)
                        annotated_image = self._render_thumbnail(
                            annotated_image, temp_manual_mask, self.current_class_idx, pad_top, is_preview=True
                        )

                    if kept_indices:
                        selected_detection = display_predictions[kept_indices][selected_idx]
                        annotated_image = self._render_thumbnail(
                            annotated_image, selected_detection.mask[0], selected_detection.class_id[0], pad_top
                        )

                    cv2.imshow("Image", annotated_image)
                    key = cv2.waitKeyEx(0)

                    if key == ord("q") or key == 27:
                        manual_mode = False
                        self.mouse_points.clear()
                        break

                    if key == ord("s"):
                        if kept_indices:
                            final_predictions = display_predictions[kept_indices]
                            original_predictions = detections[kept_indices]
                            self.save_data(cv_image, original_predictions)
                            print(f"Saved image with {len(final_predictions)} masks.")
                        else:
                            print("No masks to save.")
                        break

                    if key == ord("d") and kept_indices:
                        kept_indices.pop(selected_idx)
                        if not kept_indices:
                            selected_idx = 0
                        else:
                            selected_idx %= len(kept_indices)
                        continue

                    if not kept_indices:
                        continue

                    if key == 65364:
                        selected_idx = (selected_idx + 1) % len(kept_indices)
                    elif key == 65362:
                        selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)

                    if key == ord("m"):
                        manual_mode = not manual_mode
                        self.mouse_points.clear()
                        temp_manual_mask = None
                        print("Manual mode:", "ON" if manual_mode else "OFF")

                    if manual_mode:
                        manual_key = cv2.waitKeyEx(0)
                        if manual_key == 13:
                            temp_manual_mask = self._manual_segmentation(cv_image)
                        elif manual_key == ord("a") and temp_manual_mask is not None:
                            new_det = sv.Detections(
                                xyxy=np.array([[0, 0, cv_image.shape[1], cv_image.shape[0]]]),
                                mask=np.array([temp_manual_mask]),
                                confidence=np.array([1.0]),
                                class_id=np.array([self.current_class_idx]),
                            )
                            display_predictions += new_det
                            kept_indices.append(len(display_predictions) - 1)
                            temp_manual_mask = None
                            print("Added manual mask.")
                        elif manual_key == 65364:
                            self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                            print(f"Current class: {self.class_names[self.current_class_idx]}")
                        elif manual_key == 65362:
                            self.current_class_idx = (self.current_class_idx - 1 + len(self.class_names)) % len(
                                self.class_names
                            )
                            print(f"Current class: {self.class_names[self.current_class_idx]}")
                        elif manual_key == ord("m"):
                            manual_mode = False
                            self.mouse_points.clear()
                            temp_manual_mask = None
                        elif manual_key == 27:
                            manual_mode = False
                            self.mouse_points.clear()
                            temp_manual_mask = None
                        continue

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("RealSense camera pipeline stopped.")


__all__ = ["RealSenseSegCollector", "pre_cache_tokenizer"]
