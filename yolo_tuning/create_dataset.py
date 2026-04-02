import cv2
import numpy as np
import os
import json
import pyrealsense2 as rs
from dotenv import load_dotenv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from transformers import AutoTokenizer
import torch
import supervision as sv
import copy

# Pre-download the tokenizer to avoid network issues during initialization
def pre_cache_tokenizer():
    """Downloads and caches the tokenizer model from Hugging Face."""
    print("Pre-caching tokenizer model 'bert-base-uncased'...")
    try:
        AutoTokenizer.from_pretrained("bert-base-uncased")
        print("Tokenizer is cached.")
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
        print("Please check your internet connection and firewall settings.")
        print("You may need to configure HTTP/HTTPS proxies if you are behind a firewall.")

# Load environment variables from .env file
load_dotenv()


class TkinterBBoxCollector:
    """Tkinter-based bounding box collector with dual windows.

    - Live window: real-time camera feed
    - Review window: detections with interactive controls
    """

    def __init__(self, dataset_creator):
        self.creator = dataset_creator
        self.class_names = dataset_creator.ontology.classes() if dataset_creator.ontology else []

        from yolo_tuning.vision_tuning.data_collection.tkinter_gui import DualWindowBBoxCollector as _DualWindowCollector
        self._collector_class = _DualWindowCollector

    def start(self):
        self._collector = self._collector_class(
            class_names=self.class_names,
            on_save=self.creator.save_data,
        )
        self._collector.start()

    def update_review(self, frame, predictions):
        """Update review window with detection results."""
        return self._collector.update_review(frame, predictions)

    def is_alive(self):
        return self._collector.is_alive()

    def stop(self):
        self._collector.stop()


class RealSenseDatasetCreator:
    def __init__(self, output_dir=None, ontology_path=None, device=None):
        print("Initializing RealSenseDatasetCreator...")
        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the base model for labeling
        self.ontology_path = ontology_path or os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
        self.ontology = self._load_ontology()
        if self.ontology:
            self.base_model = GroundingDINO(ontology=self.ontology)
            if self.device.type == 'cuda':
                try:
                    # This attribute access is brittle and depends on autodistill-groundingdino implementation
                    self.base_model.dino_model.model.to(self.device)
                    self.base_model.dino_model.device = self.device
                    print("Moved GroundingDINO model to CUDA device.")
                except AttributeError:
                    print("Could not move model to CUDA. It might not be supported by this version of autodistill-groundingdino.")
        else:
            self.base_model = None

        print("Base model loaded.")
        print("Starting RealSense camera pipeline...")

        # Configure and start RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        print("RealSense camera pipeline started.")

        # Output directory setup
        self.output_dir = output_dir or os.getenv("DATASET_DIR", "dataset")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def _load_ontology(self):
        """Loads the ontology from a JSON file."""
        print("Loading ontology...")
        try:
            with open(self.ontology_path, 'r') as f:
                ontology_data = json.load(f)
            print(f"Loaded ontology from {self.ontology_path}")
            return CaptionOntology(ontology_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load ontology file: {e}. Aborting.")
            return None

    def run(self):
        """Main loop to capture, label, and save images using Tkinter GUI."""
        if not self.base_model:
            return

        print("\nStarting dataset creation with Tkinter GUI...")
        print("--- Controls ---")
        print(" ↑/↓: Select detection | 'd': Delete selected | 's': Save | Space: Skip | 'q': Quit")

        # Use Tkinter collector
        collector = TkinterBBoxCollector(self)
        collector.start()

        try:
            while collector.is_alive():
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                cv_image = np.asanyarray(color_frame.get_data())

                # Get predictions (live preview is handled by GUI internally)
                predictions = self.base_model.predict(cv_image)

                # Update review window with detection results
                if not collector.update_review(cv_image, predictions):
                    break

        finally:
            collector.stop()
            self.pipeline.stop()
            print("RealSense camera pipeline stopped.")

    def save_data(self, image, predictions):
        """Saves the image and its corresponding YOLO format labels."""
        filename_base = f"image_{len(os.listdir(self.images_dir)):05d}"
        image_filename = os.path.join(self.images_dir, f"{filename_base}.jpg")
        label_filename = os.path.join(self.labels_dir, f"{filename_base}.txt")

        cv2.imwrite(image_filename, image)

        with open(label_filename, 'w') as f:
            for box, class_id in zip(predictions.xyxy, predictions.class_id):
                x1, y1, x2, y2 = box
                h, w, _ = image.shape
                x_center = (x1 + x2) / (2 * w)
                y_center = (y1 + y2) / (2 * h)
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def main():
    load_dotenv()
    pre_cache_tokenizer()
    creator = RealSenseDatasetCreator()
    creator.run()

if __name__ == '__main__':
    main()
