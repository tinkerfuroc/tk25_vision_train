import os
import shutil
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def split_dataset(dataset_path, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        dataset_path (str): The path to the 'dataset' directory.
        train_ratio (float): The ratio of images to be used for training.
    """
    images_source_dir = os.path.join(dataset_path, "images")
    labels_source_dir = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_source_dir) or not os.path.isdir(labels_source_dir):
        print("Source 'images' and 'labels' directories not found. Exiting.")
        return

    # Define destination directories
    train_images_dest_dir = os.path.join(images_source_dir, "train")
    val_images_dest_dir = os.path.join(images_source_dir, "val")
    train_labels_dest_dir = os.path.join(labels_source_dir, "train")
    val_labels_dest_dir = os.path.join(labels_source_dir, "val")

    # Create destination directories
    os.makedirs(train_images_dest_dir, exist_ok=True)
    os.makedirs(val_images_dest_dir, exist_ok=True)
    os.makedirs(train_labels_dest_dir, exist_ok=True)
    os.makedirs(val_labels_dest_dir, exist_ok=True)

    # Get all image filenames (without extension)
    image_files = [f.split('.')[0] for f in os.listdir(images_source_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)

    # Split files
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Splitting dataset: {len(train_files)} training images, {len(val_files)} validation images.")

    # Function to move files
    def move_files(files, img_dest, lbl_dest):
        for file_base in files:
            img_filename = f"{file_base}.jpg" # Assumes jpg, adjust if needed
            lbl_filename = f"{file_base}.txt"

            shutil.move(os.path.join(images_source_dir, img_filename), os.path.join(img_dest, img_filename))
            shutil.move(os.path.join(labels_source_dir, lbl_filename), os.path.join(lbl_dest, lbl_filename))

    # Move files to their new directories
    move_files(train_files, train_images_dest_dir, train_labels_dest_dir)
    move_files(val_files, val_images_dest_dir, val_labels_dest_dir)
    
    print("Dataset splitting complete.")

if __name__ == '__main__':
    # Assuming this script is run from the root of the vision_tuning package
    # or the workspace root.
    dataset_dir = os.getenv("DATASET_DIR", "dataset")
    if os.path.exists(dataset_dir):
        split_dataset(dataset_dir)
    else:
        print(f"Dataset directory '{dataset_dir}' not found.") 