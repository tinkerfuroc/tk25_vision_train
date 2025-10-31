import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1" # Commented out to allow model downloads
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow warnings
import cv2
import numpy as np
import json
import pyrealsense2 as rs
from dotenv import load_dotenv
from lang_sam import LangSAM
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import torch
import supervision as sv
from thefuzz import process
import copy

# Pre-download the tokenizer to avoid network issues during initialization
def pre_cache_tokenizer():
    """Downloads and caches the tokenizer model from Hugging Face."""
    
    print("Pre-caching tokenizer model 'bert-base-uncased'...")
    try:
        from transformers import AutoTokenizer
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
        self.class_names = list(self.ontology.keys()) if self.ontology else []
        
        print("Loading LangSAM model...")
        self.base_model = LangSAM()
        print("LangSAM model loaded.")

        print("Loading SAM model for manual annotation...")
        # TODO: User might need to change the model type and checkpoint path
        sam_type = "vit_b"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_sam_path = os.path.join(script_dir, "..", "sam_vit_b_01ec64.pth")
        sam_checkpoint = os.getenv("SAM_CHECKPOINT_PATH", default_sam_path)
        if not os.path.exists(sam_checkpoint):
            print(f"SAM checkpoint not found at {sam_checkpoint}. Please download it or update the path in your .env file.")
            self.sam_predictor = None
        else:
            sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM model loaded.")

        print("Starting RealSense camera pipeline...")

        # Configure and start RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.pipeline.start(config)
        print("RealSense camera pipeline started.")

        # Output directory setup
        self.output_dir = os.getenv("DATASET_DIR", "dataset_seg")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Mouse callback state
        self.mouse_points = []
        self.current_class_idx = 0
        self.temp_manual_mask = None # For previewing manual segmentation

    def _load_ontology(self):
        """Loads the ontology from a JSON file."""
        print("Loading ontology...")
        ontology_path = os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
        try:
            with open(ontology_path, 'r') as f:
                ontology_data = json.load(f)
            print(f"Loaded ontology from {ontology_path}")
            return ontology_data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load ontology file: {e}. Aborting.")
            return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_points.append((x, y))
            print(f"Added point: ({x}, {y})")

    def run(self):
        """Main loop to capture, label, and save images."""
        if not self.base_model or not self.sam_predictor:
            print("Models not initialized correctly. Aborting.")
            return

        print("Starting dataset creation...")
        print("--- Live Feed Controls ---")
        print(" 'space': Capture the current frame for labeling.")
        print(" 'q': Quit the application.")
        
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_position=sv.Position.BOTTOM_LEFT)
        
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        try:
            while True: # Main loop for live feed and capturing
                # --- Live Feed ---
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                live_image = np.asanyarray(color_frame.get_data())
                display_image = live_image.copy()
                cv2.putText(display_image, "Press SPACE to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Image", display_image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                
                if key != 32: # Not spacebar, continue live feed
                    continue

                # --- Frame Captured ---
                print("Frame captured. Running segmentation...")
                cv_image = live_image.copy()
                
                # LangSAM works with RGB images
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)

                all_detections = []
                text_prompts = list(self.ontology.keys())
                text_prompt = '. '.join(text_prompts)
                print("Using text prompt:", text_prompt)           
                # Call predict once with all prompts
                predictions = self.base_model.predict([pil_image], [text_prompt])
                print(predictions[0].keys())

                # The predict method returns a list of dictionaries, one for each image.
                if predictions:
                    result = predictions[0]  # We only have one image
                    masks = result.get('masks', [])
                    boxes = result.get('boxes', [])
                    scores = result.get('scores', [])
                    phrases = result.get('text_labels', [])
                    
                    if len(masks) > 0 and len(phrases) > 0:
                        valid_detections = []
                        for i, phrase in enumerate(phrases):
                            # Use fuzzy matching to find the best class name
                            match, score = process.extractOne(phrase, self.class_names)
                            
                            # Only accept matches with a certain confidence
                            if score >= 80:
                                class_id = self.class_names.index(match)
                                
                                boolean_mask = masks[i] > 0.5
                                # Create a temporary Detections object for this one detection
                                single_detection = sv.Detections(
                                    xyxy=sv.mask_to_xyxy(masks=np.array([boolean_mask])),
                                    mask=np.array([boolean_mask]),
                                    class_id=np.array([class_id]),
                                    confidence=np.array([scores[i]])
                                )
                                valid_detections.append(single_detection)

                        if valid_detections:
                            # Merge all valid detections into a single object
                            all_detections = sv.Detections.merge(valid_detections)
                            # Apply Non-Maximum Suppression
                            # nms_indices = sv.mask_non_max_suppression(all_detections.mask, threshold=0.7)
                            # all_detections = all_detections[nms_indices]


                if all_detections:
                    predictions = all_detections
                else:
                    predictions = sv.Detections.empty()

                if len(predictions) == 0:
                    print("No automatic detections found. You can add masks manually.")

                kept_indices = list(range(len(predictions)))
                selected_idx = 0
                manual_mode = False
                self.temp_manual_mask = None # Reset temp mask

                # --- Interaction Loop for Captured Frame ---
                print("\n--- Annotation Controls ---")
                print(" 's': Save approved detections.")
                print(" 'esc': Discard and return to live feed.")
                print(" 'd': Delete selected detection.")
                print(" 'm': Enter manual annotation mode.")
                print(" 'n': Cycle class for manual annotation.")
                print("--- Manual Mode (when active) ---")
                print(" 'up'/'down'/'n': Cycle through classes.")
                print(" 'enter': Preview segmentation from points.")
                print(" 'a': Accept previewed mask.")
                print(" 'c': Clear points and preview.")


                while True: 
                    display_image = cv_image.copy()
                    
                    current_predictions = predictions[kept_indices] if kept_indices else sv.Detections.empty()

                    if len(current_predictions) > 0:
                        labels = [
                            f"#{idx} {self.class_names[cid]} {conf:.2f}"
                            for idx, (cid, conf) in enumerate(zip(current_predictions.class_id, current_predictions.confidence))
                        ]
                        annotated_image = mask_annotator.annotate(scene=display_image, detections=current_predictions)
                        annotated_image = label_annotator.annotate(scene=annotated_image, detections=current_predictions, labels=labels)
                        
                        # Highlight selected detection
                        if kept_indices and selected_idx < len(current_predictions):
                            highlight_mask = np.zeros_like(annotated_image)
                            selected_mask = current_predictions.mask[selected_idx]
                            highlight_mask[selected_mask] = [0, 0, 255] # Red highlight
                            annotated_image = cv2.addWeighted(annotated_image, 0.7, highlight_mask, 0.3, 0)
                    else:
                        annotated_image = display_image

                    if manual_mode:
                        # Draw points for manual segmentation
                        for point in self.mouse_points:
                            cv2.circle(annotated_image, point, 5, (0, 255, 0), -1)
                        
                        # Draw the temporary preview mask if it exists
                        if self.temp_manual_mask is not None:
                            temp_mask_display = np.zeros_like(annotated_image)
                            temp_mask_display[self.temp_manual_mask] = [255, 0, 0] # Blue preview
                            annotated_image = cv2.addWeighted(annotated_image, 0.7, temp_mask_display, 0.3, 0)

                        # Display current class for manual annotation
                        class_text = f"Current class: {self.class_names[self.current_class_idx]}"
                        cv2.putText(annotated_image, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Image", annotated_image)
                    key = cv2.waitKeyEx(100)

                    if key == ord('q'): # Allow quitting from inner loop too
                        self.pipeline.stop()
                        cv2.destroyAllWindows()
                        return

                    elif key == ord('s'):
                        if len(current_predictions) > 0:
                            self.save_data(cv_image, current_predictions)
                            print(f"Saved image with {len(current_predictions)} detections.")
                        else:
                            print("No detections to save.")
                        break # Break interaction loop, go back to live feed

                    elif key == 27: # Esc key
                        print("Discarded annotations. Returning to live feed.")
                        break # Break interaction loop, go back to live feed

                    elif key == ord('m'):
                        manual_mode = not manual_mode
                        self.mouse_points = []
                        self.temp_manual_mask = None
                        print(f"Manual mode {'enabled' if manual_mode else 'disabled'}.")
                    
                    # --- Key handling for both modes ---

                    # Up Arrow
                    elif key == 65362: 
                        if manual_mode:
                            self.current_class_idx = (self.current_class_idx - 1 + len(self.class_names)) % len(self.class_names)
                        elif kept_indices:
                            selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)
                    
                    # Down Arrow
                    elif key == 65364:
                        if manual_mode:
                            self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                        elif kept_indices:
                            selected_idx = (selected_idx + 1) % len(kept_indices)

                    # 'n' key for cycling classes in manual mode
                    elif key == ord('n') and manual_mode:
                        self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)

                    if manual_mode:
                        if key == 13: # Enter key to preview mask
                            if self.sam_predictor and self.mouse_points:
                                print("Generating preview mask...")
                                self.sam_predictor.set_image(rgb_image)
                                input_points = np.array(self.mouse_points)
                                input_labels = np.ones(len(self.mouse_points))
                                
                                with torch.autocast(device_type=self.device.type, enabled=False):
                                    masks, _, _ = self.sam_predictor.predict(
                                        point_coords=input_points,
                                        point_labels=input_labels,
                                        multimask_output=False,
                                    )
                                self.temp_manual_mask = masks[0]
                            else:
                                print("No points selected to generate a mask.")

                        elif key == ord('a'): # 'a' to accept the previewed mask
                            if self.temp_manual_mask is not None:
                                new_detection = sv.Detections(
                                    xyxy=sv.mask_to_xyxy(masks=np.array([self.temp_manual_mask])),
                                    mask=np.array([self.temp_manual_mask]),
                                    class_id=np.array([self.current_class_idx]),
                                    confidence=np.array([1.0]) # Manual annotation has confidence 1.0
                                )
                                
                                # Rebuild predictions from the currently kept ones, then add the new one
                                current_kept_predictions = predictions[kept_indices] if kept_indices else sv.Detections.empty()
                                predictions = sv.Detections.merge([current_kept_predictions, new_detection])
                                
                                # Reset kept_indices to reflect the new, merged predictions object
                                kept_indices = list(range(len(predictions)))
                                selected_idx = len(kept_indices) - 1 # Select the newly added mask
                                self.mouse_points = []
                                self.temp_manual_mask = None
                                print("Accepted and added new mask.")
                            else:
                                print("No preview mask to accept. Press 'Enter' to generate one first.")

                        elif key == ord('c'): # Clear points and preview
                            self.mouse_points = []
                            self.temp_manual_mask = None
                            print("Cleared points and preview.")
                        
                        elif key == 27: # Exit manual mode with Esc
                            manual_mode = False
                            self.mouse_points = []
                            self.temp_manual_mask = None
                            print("Exited manual mode.")
                        continue

                    if not kept_indices:
                        continue

                    elif key == ord('d'):
                        if kept_indices:
                            original_index = kept_indices.pop(selected_idx)
                            
                            # To properly delete, we rebuild the predictions object from the kept indices
                            predictions = predictions[kept_indices]
                            # After rebuilding, all indices are valid again
                            kept_indices = list(range(len(predictions)))
                            
                            if not kept_indices:
                                print("All detections deleted.")
                            elif selected_idx >= len(kept_indices):
                                selected_idx = len(kept_indices) - 1
                            print(f"Deleted detection.")
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("RealSense camera pipeline stopped.")

    def save_data(self, image, predictions):
        """Saves the image and its corresponding YOLO segmentation format labels."""
        filename_base = f"image_{len(os.listdir(self.images_dir)):05d}"
        image_filename = os.path.join(self.images_dir, f"{filename_base}.jpg")
        label_filename = os.path.join(self.labels_dir, f"{filename_base}.txt")

        cv2.imwrite(image_filename, image)

        h, w, _ = image.shape

        with open(label_filename, 'w') as f:
            for i in range(len(predictions)):
                class_id = predictions.class_id[i]
                mask = predictions.mask[i]
                
                # Find contours and convert to YOLO segmentation format
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue

                # Assuming the largest contour is the object
                contour = max(contours, key=cv2.contourArea)
                
                if contour.shape[0] < 3: # Need at least 3 points for a polygon
                    continue

                segment = contour.flatten().tolist()
                
                # Normalize coordinates
                normalized_segment = [val / w if i % 2 == 0 else val / h for i, val in enumerate(segment)]
                
                f.write(f"{class_id} " + " ".join(map(str, normalized_segment)) + "\n")

def main():
    load_dotenv()
    pre_cache_tokenizer()
    creator = RealSenseDatasetCreator()
    creator.run()

if __name__ == '__main__':
    main()
