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
from collections import deque

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

class RealSenseVideoDatasetCreator:
    def __init__(self):
        print("Initializing RealSenseVideoDatasetCreator...")
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the base model for labeling
        self.ontology = self._load_ontology()
        # Keys are prompts for LangSAM, values are class names for display/training
        self.prompts = list(self.ontology.keys()) if self.ontology else []
        self.class_names = list(self.ontology.values()) if self.ontology else []
        # Create mapping from prompt to class name
        self.prompt_to_class = self.ontology if self.ontology else {}
        
        print("Loading LangSAM model...")
        self.base_model = LangSAM()
        print("LangSAM model loaded.")

        print("Loading SAM model for tracking and manual annotation...")
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
        self.output_dir = os.getenv("DATASET_DIR", "dataset_seg_video")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Mouse callback state
        self.mouse_points = []
        self.current_class_idx = 0
        self.temp_manual_mask = None # For previewing manual segmentation
        
        # Video recording state
        self.is_recording = False
        self.recorded_frames = []
        self.max_frames = 50  # Maximum frames to record (10 seconds at 30fps)

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
            # Adjust mouse coordinates to be relative to the original image, not the padded display
            top, left = 50, 50 # Must match padding defined in run()
            
            # Only register clicks within the actual image area
            if x >= left and y >= top:
                adjusted_x = x - left
                adjusted_y = y - top
                self.mouse_points.append((adjusted_x, adjusted_y))
                print(f"Added point: ({adjusted_x}, {adjusted_y})")

    def _create_padded_masks(self, masks, image_shape, top, bottom, left, right):
        """Create padded versions of masks for display."""
        padded_masks = []
        for mask in masks:
            padded_mask = np.zeros((image_shape[0] + top + bottom, image_shape[1] + left + right), dtype=bool)
            padded_mask[top:top+image_shape[0], left:left+image_shape[1]] = mask
            padded_masks.append(padded_mask)
        return np.array(padded_masks)

    def _prepare_predictions_for_display(self, predictions, left, top, image_shape, padding):
        """Prepare predictions for display by offsetting and padding."""
        predictions_for_display = copy.deepcopy(predictions)
        predictions_for_display.xyxy += np.array([left, top, left, top])
        
        top_pad, bottom_pad, left_pad, right_pad = padding
        padded_masks = self._create_padded_masks(
            predictions_for_display.mask, 
            image_shape, 
            top_pad, bottom_pad, left_pad, right_pad
        )
        predictions_for_display.mask = padded_masks
        
        return predictions_for_display

    def _create_labels(self, predictions, prefix=""):
        """Create label strings for predictions."""
        labels = [
            f"{prefix}#{idx} {self.class_names[cid]} {conf:.2f}"
            for idx, (cid, conf) in enumerate(zip(predictions.class_id, predictions.confidence))
        ]
        return labels

    def _render_thumbnail(self, annotated_image, mask, class_id, top, is_preview=False):
        """Render a thumbnail of the selected/preview segment in the top padding."""
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            return annotated_image
        
        y1, x1 = mask_coords.min(axis=0)
        y2, x2 = mask_coords.max(axis=0)
        crop_h, crop_w = y2 - y1, x2 - x1
        
        if crop_h <= 0 or crop_w <= 0:
            return annotated_image
        
        # Extract the cropped region
        cropped_region = annotated_image[y1:y2, x1:x2].copy()
        
        # Calculate size for the thumbnail
        max_thumb_h = top - 10
        scale = min(max_thumb_h / crop_h, 150 / crop_w, 1.0)
        thumb_w = int(crop_w * scale)
        thumb_h = int(crop_h * scale)
        
        if thumb_h > 0 and thumb_w > 0:
            thumbnail = cv2.resize(cropped_region, (thumb_w, thumb_h))
            
            # Place thumbnail in top-right corner
            thumb_x = annotated_image.shape[1] - thumb_w - 10
            thumb_y = 5
            annotated_image[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = thumbnail
            
            # Draw border around thumbnail
            color = (255, 0, 0) if is_preview else (0, 0, 255)
            cv2.rectangle(annotated_image, (thumb_x, thumb_y), (thumb_x+thumb_w, thumb_y+thumb_h), color, 2)
        
        return annotated_image

    def _run_automatic_segmentation(self, cv_image):
        """Run LangSAM to get automatic segmentation predictions."""
        # LangSAM works with RGB images
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Use prompts (keys) for LangSAM detection
        text_prompt = '. '.join(self.prompts)
        print("Using text prompt:", text_prompt)
        
        # Call predict once with all prompts
        predictions = self.base_model.predict([pil_image], [text_prompt])
        
        all_detections = []
        if predictions:
            result = predictions[0]  # We only have one image
            masks = result.get('masks', [])
            boxes = result.get('boxes', [])
            scores = result.get('scores', [])
            phrases = result.get('text_labels', [])
            
            if len(masks) > 0 and len(phrases) > 0:
                valid_detections = []
                for i, phrase in enumerate(phrases):
                    # Use fuzzy matching to find the best prompt match
                    match, score = process.extractOne(phrase, self.prompts)
                    
                    # Only accept matches with a certain confidence
                    if score >= 80:
                        # Get the class name (value) from the matched prompt (key)
                        class_name = self.prompt_to_class[match]
                        class_id = self.class_names.index(class_name)
                        
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
        
        if all_detections:
            return all_detections, rgb_image
        else:
            return sv.Detections.empty(), rgb_image

    def _track_masks_in_frame(self, rgb_image, previous_masks, previous_boxes, class_ids):
        """Track masks from previous frame to current frame using SAM."""
        if not self.sam_predictor or len(previous_masks) == 0:
            return None
        
        self.sam_predictor.set_image(rgb_image)
        
        tracked_masks = []
        tracked_boxes = []
        valid_class_ids = []
        
        for i, (prev_mask, prev_box, class_id) in enumerate(zip(previous_masks, previous_boxes, class_ids)):
            try:
                # Use the previous box as input to SAM
                with torch.autocast(device_type=self.device.type, enabled=False):
                    masks, _, _ = self.sam_predictor.predict(
                        box=prev_box,
                        multimask_output=False,
                    )
                
                if masks is not None and len(masks) > 0:
                    new_mask = masks[0]
                    new_box = sv.mask_to_xyxy(masks=np.array([new_mask]))[0]
                    
                    tracked_masks.append(new_mask)
                    tracked_boxes.append(new_box)
                    valid_class_ids.append(class_id)
            except Exception as e:
                print(f"Failed to track mask {i}: {e}")
                continue
        
        if tracked_masks:
            return sv.Detections(
                xyxy=np.array(tracked_boxes),
                mask=np.array(tracked_masks),
                class_id=np.array(valid_class_ids),
                confidence=np.ones(len(tracked_masks))
            )
        else:
            return sv.Detections.empty()

    def _render_regular_mode(self, display_image, current_predictions, selected_idx, kept_indices, 
                            mask_annotator, label_annotator, padding, original_image_shape):
        """Render the display for regular (non-manual) mode."""
        top, bottom, left, right = padding
        
        if len(current_predictions) == 0:
            return display_image
        
        predictions_for_display = self._prepare_predictions_for_display(
            current_predictions, left, top, original_image_shape, padding
        )
        
        labels = self._create_labels(predictions_for_display)
        
        # If there's a selected detection, render it separately on top
        if kept_indices and selected_idx < len(predictions_for_display):
            # First, render all non-selected detections
            other_indices = [i for i in range(len(predictions_for_display)) if i != selected_idx]
            if other_indices:
                other_predictions = predictions_for_display[other_indices]
                other_labels = [labels[i] for i in other_indices]
                annotated_image = mask_annotator.annotate(scene=display_image, detections=other_predictions)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=other_predictions, labels=other_labels)
            else:
                annotated_image = display_image.copy()
            
            # Get the selected mask
            selected_mask = predictions_for_display.mask[selected_idx]
            
            # Create an expanded mask to clear labels near the selection
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(selected_mask.astype(np.uint8), kernel, iterations=3).astype(bool)
            
            # Reset the area of the selection to the base image (clears overlapping labels)
            annotated_image[expanded_mask] = display_image[expanded_mask]
            
            # Now render the selected detection on top with highlight
            selected_detection = predictions_for_display[selected_idx]
            selected_label_text = [labels[selected_idx]]
            
            # Create highlight layer
            highlight_mask = np.zeros_like(annotated_image)
            highlight_mask[selected_mask] = [0, 0, 255]  # Red highlight
            annotated_image = cv2.addWeighted(annotated_image, 0.7, highlight_mask, 0.3, 0)
            
            # Render the selected detection mask and label on top
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=selected_detection)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=selected_detection, labels=selected_label_text)
            
            # Add thumbnail and top label
            annotated_image = self._render_thumbnail(annotated_image, selected_mask, 
                                                     predictions_for_display.class_id[selected_idx], 
                                                     top, is_preview=False)
            
            selected_class_id = predictions_for_display.class_id[selected_idx]
            selected_conf = predictions_for_display.confidence[selected_idx]
            top_label = f"Selected: #{selected_idx} {self.class_names[selected_class_id]} {selected_conf:.2f}"
            cv2.putText(annotated_image, top_label, (left + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # No selection, render all normally
            annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
        
        return annotated_image

    def _render_manual_mode(self, display_image, current_predictions, mask_annotator, 
                           label_annotator, padding, original_image_shape):
        """Render the display for manual annotation mode."""
        top, bottom, left, right = padding
        annotated_image = display_image.copy()
        
        # Draw the temporary preview mask if it exists
        if self.temp_manual_mask is not None:
            padded_temp_mask = np.zeros((original_image_shape[0] + top + bottom, 
                                        original_image_shape[1] + left + right), dtype=bool)
            padded_temp_mask[top:top+original_image_shape[0], left:left+original_image_shape[1]] = self.temp_manual_mask
            
            # Render existing detections first, but their masks/labels will be blocked by preview
            if len(current_predictions) > 0:
                predictions_for_display = self._prepare_predictions_for_display(
                    current_predictions, left, top, original_image_shape, padding
                )
                labels = self._create_labels(predictions_for_display)
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=predictions_for_display)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
            
            # Now draw the preview mask on top, clearing any overlapping content
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(padded_temp_mask.astype(np.uint8), kernel, iterations=3).astype(bool)
            
            # Reset the area of the preview to the base image
            annotated_image[expanded_mask] = display_image[expanded_mask]
            
            # Create a mask overlay
            temp_mask_display = np.zeros_like(annotated_image)
            temp_mask_display[padded_temp_mask] = [255, 0, 0]  # Blue preview
            
            # Overlay the preview mask
            annotated_image = cv2.addWeighted(annotated_image, 0.7, temp_mask_display, 0.3, 0)
            
            # Create a single detection for the preview mask to get proper label rendering
            preview_detection = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=np.array([padded_temp_mask])),
                mask=np.array([padded_temp_mask]),
                class_id=np.array([self.current_class_idx]),
                confidence=np.array([1.0])
            )
            preview_label = [f"Preview: {self.class_names[self.current_class_idx]}"]
            
            # Render the preview label on top of everything
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=preview_detection, labels=preview_label)
        else:
            # No preview, just render existing detections normally
            if len(current_predictions) > 0:
                predictions_for_display = self._prepare_predictions_for_display(
                    current_predictions, left, top, original_image_shape, padding
                )
                labels = self._create_labels(predictions_for_display)
                annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
        
        # Draw points for manual segmentation (always on top)
        for point in self.mouse_points:
            padded_point = (point[0] + left, point[1] + top)
            cv2.circle(annotated_image, padded_point, 5, (0, 255, 0), -1)
        
        # Display thumbnail and label for preview or show current class
        if self.temp_manual_mask is not None:
            preview_label_text = f"Preview: {self.class_names[self.current_class_idx]}"
            
            # Recreate padded_temp_mask for thumbnail extraction
            padded_temp_mask_for_thumb = np.zeros((original_image_shape[0] + top + bottom, 
                                                   original_image_shape[1] + left + right), dtype=bool)
            padded_temp_mask_for_thumb[top:top+original_image_shape[0], left:left+original_image_shape[1]] = self.temp_manual_mask
            
            annotated_image = self._render_thumbnail(annotated_image, padded_temp_mask_for_thumb, 
                                                    self.current_class_idx, top, is_preview=True)
            cv2.putText(annotated_image, preview_label_text, (left + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # Display current class for manual annotation when no preview
            class_text = f"Current class: {self.class_names[self.current_class_idx]}"
            cv2.putText(annotated_image, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_image

    def _handle_manual_mode_keys(self, key, rgb_image, predictions, kept_indices):
        """Handle key presses in manual mode."""
        if key == 13:  # Enter key to preview mask
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
        
        elif key == ord('a'):  # Accept the previewed mask
            if self.temp_manual_mask is not None:
                new_detection = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=np.array([self.temp_manual_mask])),
                    mask=np.array([self.temp_manual_mask]),
                    class_id=np.array([self.current_class_idx]),
                    confidence=np.array([1.0])
                )
                
                # Rebuild predictions from the currently kept ones, then add the new one
                current_kept_predictions = predictions[kept_indices] if kept_indices else sv.Detections.empty()
                predictions = sv.Detections.merge([current_kept_predictions, new_detection])
                
                # Reset kept_indices to reflect the new, merged predictions object
                kept_indices = list(range(len(predictions)))
                selected_idx = len(kept_indices) - 1
                self.mouse_points = []
                self.temp_manual_mask = None
                print("Accepted and added new mask.")
                return predictions, kept_indices, selected_idx
            else:
                print("No preview mask to accept. Press 'Enter' to generate one first.")
        
        elif key == ord('c'):  # Clear points and preview
            self.mouse_points = []
            self.temp_manual_mask = None
            print("Cleared points and preview.")
        
        elif key == 27:  # Exit manual mode with Esc
            return None  # Signal to exit manual mode
        
        return predictions, kept_indices, None

    def _annotate_first_frame(self, first_frame, mask_annotator, label_annotator, padding):
        """Annotate the first frame of the video using LangSAM and manual tools."""
        print("\n=== Annotating First Frame ===")
        print("Running automatic segmentation...")
        
        # Run automatic segmentation
        predictions, rgb_image = self._run_automatic_segmentation(first_frame)
        
        if len(predictions) == 0:
            print("No automatic detections found. You can add masks manually.")
        
        kept_indices = list(range(len(predictions)))
        selected_idx = 0
        manual_mode = False
        self.temp_manual_mask = None
        
        top, bottom, left, right = padding
        
        # Print controls
        print("\n--- First Frame Annotation Controls ---")
        print(" 's': Accept annotations and start tracking.")
        print(" 'esc': Discard and return to live feed.")
        print(" 'd': Delete selected detection.")
        print(" 'm': Enter manual annotation mode.")
        print(" 'up'/'down': Navigate through detections.")
        print("--- Manual Mode (when active) ---")
        print(" 'up'/'down'/'n': Cycle through classes.")
        print(" 'enter': Preview segmentation from points.")
        print(" 'a': Accept previewed mask.")
        print(" 'c': Clear points and preview.")
        
        # Interaction loop for the first frame
        while True:
            # Add padding to the base image
            display_image = cv2.copyMakeBorder(first_frame, top, bottom, left, right, 
                                              cv2.BORDER_CONSTANT, value=[50, 50, 50])
            
            current_predictions = predictions[kept_indices] if kept_indices else sv.Detections.empty()
            
            # Render based on mode
            if manual_mode:
                annotated_image = self._render_manual_mode(
                    display_image, current_predictions, mask_annotator, 
                    label_annotator, padding, first_frame.shape
                )
            else:
                annotated_image = self._render_regular_mode(
                    display_image, current_predictions, selected_idx, kept_indices,
                    mask_annotator, label_annotator, padding, first_frame.shape
                )
            
            # Add instruction text
            cv2.putText(annotated_image, "First Frame - Press 's' to start tracking", 
                       (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Video Annotation", annotated_image)
            key = cv2.waitKeyEx(100)
            
            # Handle common keys
            if key == ord('q'):
                return None
            elif key == ord('s'):
                if len(current_predictions) > 0:
                    print(f"Starting tracking with {len(current_predictions)} detections.")
                    return current_predictions
                else:
                    print("No detections to track. Add at least one annotation.")
            elif key == 27:  # Esc key
                print("Discarded annotations. Returning to live feed.")
                return None
            elif key == ord('m'):
                manual_mode = not manual_mode
                self.mouse_points = []
                self.temp_manual_mask = None
                print(f"Manual mode {'enabled' if manual_mode else 'disabled'}.")
            
            # Mode-specific key handling
            if manual_mode:
                # Up/Down arrows for class cycling
                if key == 65362:  # Up arrow
                    self.current_class_idx = (self.current_class_idx - 1 + len(self.class_names)) % len(self.class_names)
                elif key == 65364:  # Down arrow
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                elif key == ord('n'):
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                
                # Handle manual mode specific keys
                result = self._handle_manual_mode_keys(key, rgb_image, predictions, kept_indices)
                if result is None:  # Exit manual mode
                    manual_mode = False
                    self.mouse_points = []
                    self.temp_manual_mask = None
                    print("Exited manual mode.")
                elif result != (predictions, kept_indices, None):
                    predictions, kept_indices, new_selected = result
                    if new_selected is not None:
                        selected_idx = new_selected
            else:
                # Regular mode navigation
                if not kept_indices:
                    continue
                
                if key == 65362:  # Up arrow
                    selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)
                elif key == 65364:  # Down arrow
                    selected_idx = (selected_idx + 1) % len(kept_indices)
                elif key == ord('d'):  # Delete
                    if kept_indices:
                        kept_indices.pop(selected_idx)
                        predictions = predictions[kept_indices]
                        kept_indices = list(range(len(predictions)))
                        
                        if not kept_indices:
                            print("All detections deleted.")
                        elif selected_idx >= len(kept_indices):
                            selected_idx = len(kept_indices) - 1
                        print("Deleted detection.")

    def _interactive_tracking_pause(self, frame_annotations, current_idx, mask_annotator, label_annotator, padding):
        """Interactive pause mode during tracking - allows navigation and editing."""
        print(f"\n--- Paused at Frame {current_idx + 1} ---")
        print("Pause Controls:")
        print(" 'left'/'right': Navigate to previous/next frames")
        print(" 'm': Edit current frame")
        print(" 'c': Continue tracking from current frame")
        print(" 'esc': Cancel and return to tracking from original pause point")
        
        top, bottom, left, right = padding
        pause_idx = current_idx
        
        while True:
            # Get current frame and annotations
            current_frame = self.recorded_frames[pause_idx]
            current_predictions = frame_annotations[pause_idx]
            
            # Render frame
            display_image = cv2.copyMakeBorder(current_frame, top, bottom, left, right, 
                                              cv2.BORDER_CONSTANT, value=[50, 50, 50])
            
            if len(current_predictions) > 0:
                predictions_for_display = self._prepare_predictions_for_display(
                    current_predictions, left, top, current_frame.shape, padding
                )
                labels = self._create_labels(predictions_for_display)
                
                annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
            else:
                annotated_image = display_image.copy()
            
            # Add status text
            cv2.putText(annotated_image, f"PAUSED - Frame {pause_idx + 1}/{len(frame_annotations)} (Navigate: left/right, Edit: m, Continue: c)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            cv2.imshow("Video Annotation", annotated_image)
            
            # Get user input
            pause_key = cv2.waitKeyEx(100)
            
            if pause_key == ord('c'):  # Continue tracking from current frame
                print(f"Continuing tracking from frame {pause_idx + 1}...")
                return pause_idx  # Return the frame index to continue from
            elif pause_key == 27:  # Esc - cancel, go back to original pause point
                print("Cancelled. Returning to original pause point...")
                return current_idx  # Return original index
            elif pause_key == 65363:  # Right arrow
                if pause_idx < len(frame_annotations) - 1:
                    pause_idx += 1
            elif pause_key == 65361:  # Left arrow
                if pause_idx > 0:
                    pause_idx -= 1
            elif pause_key == ord('m'):  # Edit current frame
                print(f"\nEditing frame {pause_idx + 1}...")
                edited_predictions = self._edit_single_frame(
                    current_frame, current_predictions,
                    mask_annotator, label_annotator, padding
                )
                
                if edited_predictions is not None:
                    frame_annotations[pause_idx] = edited_predictions
                    print(f"Frame {pause_idx + 1} updated.")
                else:
                    print("Edit cancelled.")

    def _track_video(self, initial_predictions, mask_annotator, label_annotator, padding):
        """Track the annotations through the rest of the video frames with interactive pause."""
        print("\n=== Tracking Through Video ===")
        print(f"Total frames recorded: {len(self.recorded_frames)}")
        print("Press 'p' to pause (navigate/edit), 'esc' to cancel tracking")
        
        # Store annotations for each frame
        frame_annotations = [initial_predictions]  # First frame
        
        top, bottom, left, right = padding
        
        # Track through remaining frames
        i = 1
        while i < len(self.recorded_frames):
            current_frame = self.recorded_frames[i]
            rgb_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Get previous frame's predictions
            prev_predictions = frame_annotations[i - 1]
            
            # Track masks
            tracked_predictions = self._track_masks_in_frame(
                rgb_image,
                prev_predictions.mask,
                prev_predictions.xyxy,
                prev_predictions.class_id
            )
            
            if tracked_predictions is None or len(tracked_predictions) == 0:
                print(f"Warning: Tracking failed at frame {i}. Using previous frame's predictions.")
                tracked_predictions = prev_predictions
            
            # Append or update the frame annotation
            if i < len(frame_annotations):
                frame_annotations[i] = tracked_predictions
            else:
                frame_annotations.append(tracked_predictions)
            
            # Show preview with tracking status
            display_image = cv2.copyMakeBorder(current_frame, top, bottom, left, right, 
                                              cv2.BORDER_CONSTANT, value=[50, 50, 50])
            
            predictions_for_display = self._prepare_predictions_for_display(
                tracked_predictions, left, top, current_frame.shape, padding
            )
            labels = self._create_labels(predictions_for_display)
            
            annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
            
            cv2.putText(annotated_image, f"Tracking: Frame {i + 1}/{len(self.recorded_frames)} - Press 'p' to pause", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Video Annotation", annotated_image)
            
            # Check for user input
            key = cv2.waitKeyEx(1)
            
            if key == ord('p'):  # Pause - enter interactive navigation mode
                new_start_idx = self._interactive_tracking_pause(
                    frame_annotations, i, mask_annotator, label_annotator, padding
                )
                # Continue tracking from the returned frame index
                i = new_start_idx
            elif key == 27:  # Esc - cancel tracking
                print("\nTracking cancelled by user.")
                return None
            
            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Tracked {i + 1}/{len(self.recorded_frames)} frames...")
            
            i += 1
        
        print("Tracking complete!")
        return frame_annotations

    def _review_and_edit_video(self, frame_annotations, mask_annotator, label_annotator, padding):
        """Review and manually edit the annotated video frame by frame."""
        print("\n=== Reviewing and Editing Video ===")
        print("Controls:")
        print(" 'left'/'right': Navigate frames")
        print(" 'space': Play/Pause")
        print(" 'm': Enter manual edit mode for current frame")
        print(" 't': Re-run tracking from current frame onwards")
        print(" 'x': Discard current frame (remove from video)")
        print(" 'p': Discard all frames after current frame")
        print(" 's': Save all frames")
        print(" 'esc': Discard and return")
        
        current_frame_idx = 0
        playing = False
        top, bottom, left, right = padding
        
        while True:
            current_frame = self.recorded_frames[current_frame_idx]
            current_predictions = frame_annotations[current_frame_idx]
            
            # Render frame
            display_image = cv2.copyMakeBorder(current_frame, top, bottom, left, right, 
                                              cv2.BORDER_CONSTANT, value=[50, 50, 50])
            
            if len(current_predictions) > 0:
                predictions_for_display = self._prepare_predictions_for_display(
                    current_predictions, left, top, current_frame.shape, padding
                )
                labels = self._create_labels(predictions_for_display)
                
                annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
            else:
                annotated_image = display_image.copy()
            
            # Add info text
            status = "Playing" if playing else "Paused"
            cv2.putText(annotated_image, f"Frame {current_frame_idx + 1}/{len(self.recorded_frames)} - {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_image, "s:save | m:edit | t:retrack | x:delete frame | p:prune after", 
                       (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("Video Annotation", annotated_image)
            
            # Adjust wait time based on playing status
            wait_time = 33 if playing else 100  # ~30fps when playing
            key = cv2.waitKeyEx(wait_time)
            
            if key == ord('q'):
                return None
            elif key == ord('s'):
                print("Saving all frames...")
                return frame_annotations
            elif key == 27:  # Esc
                print("Discarded edits.")
                return None
            elif key == 32:  # Space - play/pause
                playing = not playing
            elif key == 65363:  # Right arrow
                current_frame_idx = min(current_frame_idx + 1, len(self.recorded_frames) - 1)
                playing = False
            elif key == 65361:  # Left arrow
                current_frame_idx = max(current_frame_idx - 1, 0)
                playing = False
            elif key == ord('m'):  # Manual edit mode
                playing = False
                print(f"\n--- Editing Frame {current_frame_idx + 1} ---")
                edited_predictions = self._edit_single_frame(
                    current_frame, current_predictions, 
                    mask_annotator, label_annotator, padding
                )
                if edited_predictions is not None:
                    frame_annotations[current_frame_idx] = edited_predictions
                    print("Frame edits saved.")
            elif key == ord('t'):  # Re-run tracking from current frame
                playing = False
                if len(current_predictions) > 0:
                    print(f"\n--- Re-running tracking from frame {current_frame_idx + 1} ---")
                    print("Press 'p' to pause (navigate/edit), 'esc' to cancel re-tracking")
                    
                    # Track from current frame to the end (interactive)
                    i = current_frame_idx + 1
                    while i < len(self.recorded_frames):
                        frame = self.recorded_frames[i]
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Get previous frame's predictions
                        prev_predictions = frame_annotations[i - 1]
                        
                        # Track masks
                        tracked_predictions = self._track_masks_in_frame(
                            rgb_image,
                            prev_predictions.mask,
                            prev_predictions.xyxy,
                            prev_predictions.class_id
                        )
                        
                        if tracked_predictions is None or len(tracked_predictions) == 0:
                            print(f"Warning: Tracking failed at frame {i}. Using previous frame's predictions.")
                            tracked_predictions = prev_predictions
                        
                        frame_annotations[i] = tracked_predictions
                        
                        # Show preview with tracking status
                        display_image = cv2.copyMakeBorder(frame, top, bottom, left, right, 
                                                          cv2.BORDER_CONSTANT, value=[50, 50, 50])
                        
                        predictions_for_display = self._prepare_predictions_for_display(
                            tracked_predictions, left, top, frame.shape, padding
                        )
                        labels = self._create_labels(predictions_for_display)
                        
                        annotated_image = mask_annotator.annotate(scene=display_image, detections=predictions_for_display)
                        annotated_image = label_annotator.annotate(scene=annotated_image, detections=predictions_for_display, labels=labels)
                        
                        cv2.putText(annotated_image, f"Re-tracking: Frame {i + 1}/{len(self.recorded_frames)} - Press 'p' to pause", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.imshow("Video Annotation", annotated_image)
                        
                        # Check for user input
                        retrack_key = cv2.waitKeyEx(1)
                        
                        if retrack_key == ord('p'):  # Pause - enter interactive navigation mode
                            new_start_idx = self._interactive_tracking_pause(
                                frame_annotations, i, mask_annotator, label_annotator, padding
                            )
                            # Continue tracking from the returned frame index
                            i = new_start_idx
                        elif retrack_key == 27:  # Esc - cancel re-tracking
                            print("\nRe-tracking cancelled by user.")
                            break
                        
                        # Display progress
                        if (i + 1) % 5 == 0 or i == len(self.recorded_frames) - 1:
                            print(f"Re-tracked {i - current_frame_idx}/{len(self.recorded_frames) - current_frame_idx - 1} frames...")
                        
                        i += 1
                    
                    print("Re-tracking complete!")
                else:
                    print("Current frame has no annotations. Cannot track from empty frame.")
            elif key == ord('x'):  # Discard current frame entirely
                playing = False
                print(f"Deleting frame {current_frame_idx + 1}...")
                # Remove the frame and its annotations
                del self.recorded_frames[current_frame_idx]
                del frame_annotations[current_frame_idx]
                print(f"Frame deleted. Video now has {len(self.recorded_frames)} frames.")
                
                # Adjust current index if needed
                if len(self.recorded_frames) == 0:
                    print("No frames remaining. Exiting review mode.")
                    return None
                elif current_frame_idx >= len(self.recorded_frames):
                    current_frame_idx = len(self.recorded_frames) - 1
            elif key == ord('p'):  # Prune/discard all frames after current
                playing = False
                # Truncate the video at current frame
                print(f"Pruning all frames after frame {current_frame_idx + 1}...")
                self.recorded_frames = self.recorded_frames[:current_frame_idx + 1]
                frame_annotations = frame_annotations[:current_frame_idx + 1]
                print(f"Video now has {len(self.recorded_frames)} frames.")
                # Ensure we don't go out of bounds
                if current_frame_idx >= len(self.recorded_frames):
                    current_frame_idx = len(self.recorded_frames) - 1
            
            # Auto-advance if playing
            if playing:
                current_frame_idx += 1
                if current_frame_idx >= len(self.recorded_frames):
                    current_frame_idx = 0  # Loop

    def _edit_single_frame(self, frame, predictions, mask_annotator, label_annotator, padding):
        """Edit annotations for a single frame."""
        kept_indices = list(range(len(predictions)))
        selected_idx = 0
        manual_mode = False
        self.temp_manual_mask = None
        self.mouse_points = []
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        top, bottom, left, right = padding
        
        print("Frame edit mode - Controls:")
        print(" 'up'/'down': Navigate detections")
        print(" 'd': Delete selected")
        print(" 'm': Manual annotation mode")
        print(" 's': Save changes")
        print(" 'esc': Cancel changes")
        
        while True:
            display_image = cv2.copyMakeBorder(frame, top, bottom, left, right, 
                                              cv2.BORDER_CONSTANT, value=[50, 50, 50])
            
            current_predictions = predictions[kept_indices] if kept_indices else sv.Detections.empty()
            
            if manual_mode:
                annotated_image = self._render_manual_mode(
                    display_image, current_predictions, mask_annotator, 
                    label_annotator, padding, frame.shape
                )
            else:
                annotated_image = self._render_regular_mode(
                    display_image, current_predictions, selected_idx, kept_indices,
                    mask_annotator, label_annotator, padding, frame.shape
                )
            
            cv2.putText(annotated_image, "Editing Frame - 's' to save, 'esc' to cancel", 
                       (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Video Annotation", annotated_image)
            key = cv2.waitKeyEx(100)
            
            if key == ord('s'):
                return current_predictions
            elif key == 27:
                return None
            elif key == ord('m'):
                manual_mode = not manual_mode
                self.mouse_points = []
                self.temp_manual_mask = None
            
            if manual_mode:
                if key == 65362:  # Up arrow
                    self.current_class_idx = (self.current_class_idx - 1 + len(self.class_names)) % len(self.class_names)
                elif key == 65364:  # Down arrow
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                elif key == ord('n'):
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.class_names)
                
                result = self._handle_manual_mode_keys(key, rgb_image, predictions, kept_indices)
                if result is None:
                    manual_mode = False
                    self.mouse_points = []
                    self.temp_manual_mask = None
                elif result != (predictions, kept_indices, None):
                    predictions, kept_indices, new_selected = result
                    if new_selected is not None:
                        selected_idx = new_selected
            else:
                # Regular mode navigation - only navigate if we have indices
                if kept_indices:
                    if key == 65362:  # Up arrow
                        selected_idx = (selected_idx - 1 + len(kept_indices)) % len(kept_indices)
                    elif key == 65364:  # Down arrow
                        selected_idx = (selected_idx + 1) % len(kept_indices)
                    elif key == ord('d'):
                        if kept_indices:
                            kept_indices.pop(selected_idx)
                            predictions = predictions[kept_indices]
                            kept_indices = list(range(len(predictions)))
                            
                            if not kept_indices:
                                print("All detections deleted.")
                            elif selected_idx >= len(kept_indices):
                                selected_idx = len(kept_indices) - 1

    def save_video_data(self, frame_annotations):
        """Save all frames and annotations in YOLO segmentation format."""
        print(f"\nSaving {len(self.recorded_frames)} frames...")
        
        saved_count = 0
        for i, (frame, predictions) in enumerate(zip(self.recorded_frames, frame_annotations)):
            if len(predictions) == 0:
                print(f"Skipping frame {i} (no annotations)")
                continue
            
            filename_base = f"video_frame_{len(os.listdir(self.images_dir)):05d}"
            image_filename = os.path.join(self.images_dir, f"{filename_base}.jpg")
            label_filename = os.path.join(self.labels_dir, f"{filename_base}.txt")
            
            cv2.imwrite(image_filename, frame)
            
            h, w, _ = frame.shape
            
            with open(label_filename, 'w') as f:
                for j in range(len(predictions)):
                    class_id = predictions.class_id[j]
                    mask = predictions.mask[j]
                    
                    # Find contours and convert to YOLO segmentation format
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    # Assuming the largest contour is the object
                    contour = max(contours, key=cv2.contourArea)
                    
                    if contour.shape[0] < 3:  # Need at least 3 points for a polygon
                        continue
                    
                    segment = contour.flatten().tolist()
                    
                    # Normalize coordinates
                    normalized_segment = [val / w if idx % 2 == 0 else val / h for idx, val in enumerate(segment)]
                    
                    f.write(f"{class_id} " + " ".join(map(str, normalized_segment)) + "\n")
            
            saved_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Saved {i + 1}/{len(self.recorded_frames)} frames...")
        
        print(f"Successfully saved {saved_count} frames with annotations!")

    def run(self):
        """Main loop to record, annotate, and save video data."""
        if not self.base_model or not self.sam_predictor:
            print("Models not initialized correctly. Aborting.")
            return

        print("Starting video dataset creation...")
        print("\n--- Live Feed Controls ---")
        print(" 'r': Start/Stop recording video")
        print(" 'q': Quit the application")
        
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_position=sv.Position.BOTTOM_LEFT)
        
        cv2.namedWindow("Video Annotation")
        cv2.setMouseCallback("Video Annotation", self.mouse_callback)

        # Define padding for the display
        padding = (50, 100, 50, 50)  # top, bottom, left, right
        top, bottom, left, right = padding
        border_color = [50, 50, 50]  # Dark gray

        try:
            while True:  # Main loop for live feed and recording
                # --- Live Feed ---
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                live_image = np.asanyarray(color_frame.get_data())
                
                # Record frame if recording is active
                if self.is_recording:
                    self.recorded_frames.append(live_image.copy())
                    if len(self.recorded_frames) >= self.max_frames:
                        print(f"Reached maximum frames ({self.max_frames}). Stopping recording.")
                        # Stop recording and process
                        self.is_recording = False
                        print(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
                        
                        if len(self.recorded_frames) > 0:
                            # Process the recorded video
                            first_frame = self.recorded_frames[0]
                            
                            # Annotate first frame
                            initial_predictions = self._annotate_first_frame(
                                first_frame, mask_annotator, label_annotator, padding
                            )
                            
                            if initial_predictions is None:
                                print("First frame annotation cancelled.")
                                continue
                            
                            # Track through video
                            frame_annotations = self._track_video(
                                initial_predictions, mask_annotator, label_annotator, padding
                            )
                            
                            # Review and edit
                            final_annotations = self._review_and_edit_video(
                                frame_annotations, mask_annotator, label_annotator, padding
                            )
                            
                            if final_annotations is not None:
                                # Save data
                                self.save_video_data(final_annotations)
                            else:
                                print("Video annotation cancelled.")
                        else:
                            print("No frames recorded.")
                
                # Add padding for display
                display_image = cv2.copyMakeBorder(live_image, top, bottom, left, right, 
                                                  cv2.BORDER_CONSTANT, value=border_color)

                # Display status
                if self.is_recording:
                    status_text = f"RECORDING: {len(self.recorded_frames)}/{self.max_frames} frames"
                    color = (0, 0, 255)  # Red for recording
                else:
                    status_text = "Press 'r' to start recording, 'q' to quit"
                    color = (255, 255, 255)
                
                cv2.putText(display_image, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Video Annotation", display_image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not self.is_recording:
                        # Start recording
                        self.is_recording = True
                        self.recorded_frames = []
                        print("Started recording...")
                    else:
                        # Stop recording and process
                        self.is_recording = False
                        print(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
                        
                        if len(self.recorded_frames) > 0:
                            # Process the recorded video
                            first_frame = self.recorded_frames[0]
                            
                            # Annotate first frame
                            initial_predictions = self._annotate_first_frame(
                                first_frame, mask_annotator, label_annotator, padding
                            )
                            
                            if initial_predictions is None:
                                print("First frame annotation cancelled.")
                                continue
                            
                            # Track through video
                            frame_annotations = self._track_video(
                                initial_predictions, mask_annotator, label_annotator, padding
                            )
                            
                            # Review and edit
                            final_annotations = self._review_and_edit_video(
                                frame_annotations, mask_annotator, label_annotator, padding
                            )
                            
                            if final_annotations is not None:
                                # Save data
                                self.save_video_data(final_annotations)
                            else:
                                print("Video annotation cancelled.")
                        else:
                            print("No frames recorded.")
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("RealSense camera pipeline stopped.")

def main():
    load_dotenv()
    pre_cache_tokenizer()
    creator = RealSenseVideoDatasetCreator()
    creator.run()

if __name__ == '__main__':
    main()
