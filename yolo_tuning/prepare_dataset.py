import os
from dotenv import load_dotenv

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection import split_yolo_dataset as split_dataset

# Load environment variables from .env file
load_dotenv()


if __name__ == "__main__":
    config = VisionConfig()
    dataset_dir = os.getenv("DATASET_DIR", config.dataset_dir)
    if os.path.exists(dataset_dir):
        train_count, val_count = split_dataset(dataset_dir, train_ratio=0.8, seed=config.seed)
        print(f"Split dataset at {dataset_dir}: {train_count} train / {val_count} val images")
    else:
        print(f"Dataset directory '{dataset_dir}' not found.")
