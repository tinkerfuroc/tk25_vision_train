import cv2
import os
import torch
import pyrealsense2 as rs
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_live_test(model_path):
    """
    Runs a live test of a trained YOLO model using a RealSense camera.

    Args:
        model_path (str): The path to the trained YOLO model file (.pt).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first to generate the model.")
        return

    # 1. Initialize Model and Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(model_path)
    model.to(device)

    # 2. Initialize RealSense Camera
    print("Starting RealSense camera pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("RealSense camera pipeline started.")
    except Exception as e:
        print(f"Failed to start RealSense camera: {e}")
        return

    # 3. Initialize Annotators
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT)

    # 4. Main Loop
    print("Running live inference... Press 'q' to quit.")
    try:
        while True:
            # Get frame from camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert to numpy array
            cv_image = np.asanyarray(color_frame.get_data())

            # Run inference
            results = model(cv_image, verbose=False)[0]
            
            # Convert results to supervision Detections object
            detections = sv.Detections.from_ultralytics(results)

            # Generate labels
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Annotate the frame
            annotated_image = mask_annotator.annotate(scene=cv_image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # Display the frame
            cv2.imshow("Live Test", annotated_image)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")


if __name__ == '__main__':
    # Default path where the training script saves the best model
    default_model_path = os.getenv("BEST_MODEL_PATH", "yolo_finetuned_seg_best.pt")
    run_live_test(default_model_path)
