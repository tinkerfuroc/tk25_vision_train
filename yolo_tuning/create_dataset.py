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

class RealSenseDatasetCreator:
    def __init__(self):
        print("Initializing RealSenseDatasetCreator...")
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the base model for labeling
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
        self.output_dir = os.getenv("DATASET_DIR", "dataset")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def _load_ontology(self):
        """Loads the ontology from a JSON file."""
        print("Loading ontology...")
        ontology_path = os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
        try:
            with open(ontology_path, 'r') as f:
                ontology_data = json.load(f)
            print(f"Loaded ontology from {ontology_path}")
            return CaptionOntology(ontology_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load ontology file: {e}. Aborting.")
            return None

    def run(self):
        """Main loop to capture, label, and save images."""
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
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                cv_image = np.asanyarray(color_frame.get_data())

                # Define padding for display
                pad_top, pad_bottom, pad_left, pad_right = 50, 50, 50, 50
                border_color = [0, 0, 0]  # Black border

                # Use autodistill to get predictions
                predictions = self.base_model.predict(cv_image)

                if len(predictions) == 0:
                    display_image = cv2.copyMakeBorder(cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border_color)
                    cv2.imshow("Image", display_image)
                    print("No detections found. Press any key to skip, or 'q' to quit.")
                    key = cv2.waitKeyEx(0)
                    if key == ord('q'):
                        print("Quitting.")
                        break
                    else:
                        print("Skipped image (no detections).")
                        continue
                
                # Offset predictions to match the padded image for display
                display_predictions = copy.deepcopy(predictions)
                display_predictions.xyxy[:, [0, 2]] += pad_left
                display_predictions.xyxy[:, [1, 3]] += pad_top

                kept_indices = list(range(len(predictions)))
                selected_idx = 0

                while True: # Loop for interaction on a single image
                    # Create a padded image for display for this interaction loop
                    display_image = cv2.copyMakeBorder(cv_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=border_color)

                    if not kept_indices:
                        # No detections left to show
                        cv2.imshow("Image", display_image)
                    else:
                        detections_to_show = display_predictions[kept_indices]
                        
                        labels = [
                            f"{self.ontology.classes()[class_id]} {confidence:0.2f}"
                            for class_id, confidence in zip(detections_to_show.class_id, detections_to_show.confidence)
                        ]
                        
                        annotated_image = box_annotator.annotate(scene=display_image, detections=detections_to_show)
                        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections_to_show, labels=labels)

                        # Highlight selected detection
                        selected_detection = detections_to_show[selected_idx]
                        annotated_image = highlight_annotator.annotate(scene=annotated_image, detections=selected_detection)
                        
                        cv2.imshow("Image", annotated_image)

                    key = cv2.waitKeyEx(0)

                    if key == ord('q'): # Quit
                        self.pipeline.stop()
                        cv2.destroyAllWindows()
                        print("Quitting application.")
                        return

                    elif key == ord('s'): # Save
                        if kept_indices:
                            final_predictions = predictions[kept_indices]
                            self.save_data(cv_image, final_predictions)
                            print(f"Saved image with {len(final_predictions)} detections.")
                        else:
                            print("No detections to save.")
                        break 

                    elif key == 32: # Skip image with space bar
                        print("Skipped image.")
                        break 

                    if not kept_indices:
                        continue
                    
                    if key == 65364: # Next detection with down arrow
                        selected_idx = (selected_idx + 1) % len(kept_indices)
                    
                    elif key == 65362: # Previous detection with up arrow
                        selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)

                    elif key == ord('d'): # Delete
                        kept_indices.pop(selected_idx)
                        if not kept_indices:
                            print("All detections deleted.")
                            continue
                        if selected_idx >= len(kept_indices):
                            selected_idx = len(kept_indices) - 1
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()
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
