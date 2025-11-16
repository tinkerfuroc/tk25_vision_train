import os
import yaml
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from ultralytics import YOLO
from prepare_dataset import split_dataset

# Load environment variables from .env file
load_dotenv()

def plot_training_results(results_dir):
    """
    Plots training and validation losses from the results.csv file.

    Args:
        results_dir (str): The directory where the training results are saved.
    """
    results_csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(results_csv_path):
        print(f"Could not find results.csv in {results_dir}")
        return

    # Read the data
    results = pd.read_csv(results_csv_path)
    # The column names have leading spaces, so we strip them.
    results.columns = results.columns.str.strip()

    epochs = results['epoch']
    
    # Plotting for segmentation - adjust figure size and subplots
    plt.figure(figsize=(24, 10))

    # Segmentation Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, results['train/seg_loss'], label='Train Segmentation Loss')
    plt.plot(epochs, results['val/seg_loss'], label='Validation Segmentation Loss')
    plt.title('Segmentation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Class Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, results['train/cls_loss'], label='Train Class Loss')
    plt.plot(epochs, results['val/cls_loss'], label='Validation Class Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Box Loss (segmentation models still have box predictions)
    plt.subplot(2, 3, 3)
    plt.plot(epochs, results['train/box_loss'], label='Train Box Loss')
    plt.plot(epochs, results['val/box_loss'], label='Validation Box Loss')
    plt.title('Bounding Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mAP for Masks
    plt.subplot(2, 3, 4)
    plt.plot(epochs, results['metrics/mAP50-95(M)'], label='mAP50-95 (Mask)')
    plt.plot(epochs, results['metrics/mAP50(M)'], label='mAP50 (Mask)')
    plt.title('Mean Average Precision (Masks)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)

    # Confusion Matrix
    confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        plt.subplot(2, 3, 5)
        img = plt.imread(confusion_matrix_path)
        plt.imshow(img)
        plt.title('Confusion Matrix')
        plt.axis('off')  # Hide the axes for the image
    else:
        print("confusion_matrix.png not found, skipping plot.")

    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(results_dir, 'training_plots.png')
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    plt.close()


def run_finetuning():
    """
    Runs the full fine-tuning pipeline for a YOLO model.
    """
    # 1. Define Paths from .env file
    dataset_path = os.getenv("DATASET_DIR", "dataset_seg") # Changed to dataset_seg
    ontology_path = os.getenv("ONTOLOGY_PATH", "resource/ontology.json")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "runs")
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')

    # 2. Prepare the dataset
    print("--- Preparing Dataset ---")
    # Check if dataset needs splitting
    train_dir = os.path.join(dataset_path, 'images', 'train')
    if not os.path.exists(train_dir):
        split_dataset(dataset_path)
    else:
        print("Dataset already split. Skipping split.")
    print("------------------------")


    # 3. Create the data.yaml file from ontology
    print("--- Creating data.yaml ---")
    try:
        with open(ontology_path, 'r') as f:
            ontology = json.load(f)
        
        # Use ontology values (actual class names) for training
        # Keys are prompts for detection, values are class names
        class_names = list(ontology.values())
        num_classes = len(class_names)

        data_yaml = {
            'path': os.path.abspath(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': num_classes,
            'names': class_names
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"Successfully created {data_yaml_path} with {num_classes} classes.")

    except Exception as e:
        print(f"Error creating data.yaml: {e}")
        return
    print("--------------------------")

    # 4. Run the training
    print("--- Starting YOLO Training ---")
    try:
        # Load a pretrained segmentation model.
        model = YOLO('yolo11s-seg.pt')

        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=250,
            imgsz=640,
            batch=4,
            project=checkpoint_dir,
            name='yolo_finetuned'
        )
        
        print("--- Training Complete ---")
        results_dir = results.save_dir
        print(f"Model and results saved in: {results_dir}")
        
        # Plot the results
        plot_training_results(results_dir)

        # Optionally, move the best model to the root for easy access
        best_model_path = os.path.join(results_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, './yolo_seg_finetuned_best.pt')
            print(f"Best model copied to ./yolo_seg_finetuned_best.pt")

    except Exception as e:
        print(f"An error occurred during training: {e}")
    print("----------------------------")


if __name__ == '__main__':
    run_finetuning()
