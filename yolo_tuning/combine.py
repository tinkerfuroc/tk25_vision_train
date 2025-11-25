import os
import shutil

def combine_datasets(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define subdirectories for images and labels
    source_images = os.path.join(source_dir, 'images')
    source_labels = os.path.join(source_dir, 'labels')
    target_images = os.path.join(target_dir, 'images')
    target_labels = os.path.join(target_dir, 'labels')

    # Create target subdirectories
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)

    # Copy and rename images
    for filename in os.listdir(source_images):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            new_filename = f"{os.path.splitext(filename)[0]}_2{os.path.splitext(filename)[1]}"
            shutil.copy(os.path.join(source_images, filename), os.path.join(target_images, new_filename))

    # Copy and rename labels
    for filename in os.listdir(source_labels):
        if filename.endswith('.txt'):
            new_filename = f"{os.path.splitext(filename)[0]}_2.txt"
            shutil.copy(os.path.join(source_labels, filename), os.path.join(target_labels, new_filename))

if __name__ == "__main__":
    source_dataset = '/home/tinker/tk25_vision_train/yolo_tuning/dataset_zgc_new'  # Replace with the path to the dataset to be merged
    target_dataset = '/home/tinker/tk25_vision_train/yolo_tuning/dataset_zgc_new2_combined_new'  # Replace with the path to the target dataset
    combine_datasets(source_dataset, target_dataset)