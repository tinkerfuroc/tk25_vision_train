import os
import shutil
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.splitter import split_yolo_dataset
from yolo_tuning.vision_tuning.ontology import Ontology


def _prepare_dataset(dataset_path: str, train_ratio: float, seed: int) -> None:
    train_dir = os.path.join(dataset_path, "images", "train")
    if os.path.exists(train_dir):
        return
    split_yolo_dataset(dataset_path, train_ratio=train_ratio, seed=seed)


def _plot_training_results(results_dir: str, task: str) -> None:
    results_csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_csv_path):
        return

    results = pd.read_csv(results_csv_path)
    results.columns = results.columns.str.strip()
    epochs = results["epoch"]

    if task == "seg":
        metrics = [
            ("train/seg_loss", "val/seg_loss", "Segmentation Loss"),
            ("train/cls_loss", "val/cls_loss", "Classification Loss"),
            ("train/box_loss", "val/box_loss", "Bounding Box Loss"),
        ]
        maps = [
            ("metrics/mAP50-95(M)", "metrics/mAP50(M)", "Mean Average Precision (Masks)"),
        ]
    else:
        metrics = [
            ("train/box_loss", "val/box_loss", "Bounding Box Loss"),
            ("train/cls_loss", "val/cls_loss", "Classification Loss"),
            ("train/dfl_loss", "val/dfl_loss", "Distribution Focal Loss"),
        ]
        maps = [
            ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "Mean Average Precision (Boxes)"),
        ]

    plt.figure(figsize=(20, 8))
    subplot_idx = 1
    for train_col, val_col, title in metrics:
        plt.subplot(2, 3, subplot_idx)
        if train_col in results and val_col in results:
            plt.plot(epochs, results[train_col], label="Train")
            plt.plot(epochs, results[val_col], label="Validation")
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
        subplot_idx += 1

    for map_col, map50_col, title in maps:
        plt.subplot(2, 3, subplot_idx)
        if map_col in results and map50_col in results:
            plt.plot(epochs, results[map_col], label="mAP50-95")
            plt.plot(epochs, results[map50_col], label="mAP50")
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel("mAP")
            plt.legend()
            plt.grid(True)
        subplot_idx += 1

    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        plt.subplot(2, 3, subplot_idx)
        img = plt.imread(confusion_matrix_path)
        plt.imshow(img)
        plt.title("Confusion Matrix")
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(results_dir, "training_plots.png")
    plt.savefig(save_path)
    plt.close()


def _train(
    *,
    config: VisionConfig,
    dataset_path: str,
    data_yaml_path: str,
    weights: str,
    epochs: int,
    batch: int,
    imgsz: int,
    train_ratio: float,
    task: str,
    project_name: str,
) -> Tuple[str, str]:
    ontology = Ontology.from_path(config.ontology_path)
    _prepare_dataset(dataset_path, train_ratio=train_ratio, seed=config.seed)
    ontology.to_data_yaml(dataset_path, data_yaml_path)

    model = YOLO(weights)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=config.checkpoint_dir,
        name=project_name,
    )

    results_dir = results.save_dir
    _plot_training_results(results_dir, task=task)

    best_model_path = os.path.join(results_dir, "weights", "best.pt")
    exported_best = ""
    if os.path.exists(best_model_path):
        exported_best = os.path.join(os.getcwd(), f"{project_name}_best.pt")
        shutil.copy(best_model_path, exported_best)

    return exported_best, results_dir


def train_detector(
    config: VisionConfig,
    *,
    dataset_path: Optional[str] = None,
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 640,
    train_ratio: float = 0.8,
) -> Tuple[str, str]:
    dataset_root = dataset_path or config.dataset_dir
    data_yaml_path = os.path.join(dataset_root, "data.yaml")
    return _train(
        config=config,
        dataset_path=dataset_root,
        data_yaml_path=data_yaml_path,
        weights=config.detection_weights,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        train_ratio=train_ratio,
        task="det",
        project_name="yolo_finetuned",
    )


def train_segmenter(
    config: VisionConfig,
    *,
    dataset_path: Optional[str] = None,
    epochs: int = 250,
    batch: int = 4,
    imgsz: int = 640,
    train_ratio: float = 0.8,
) -> Tuple[str, str]:
    dataset_root = dataset_path or config.seg_dataset_dir
    data_yaml_path = os.path.join(dataset_root, "data.yaml")
    return _train(
        config=config,
        dataset_path=dataset_root,
        data_yaml_path=data_yaml_path,
        weights=config.segmentation_weights,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        train_ratio=train_ratio,
        task="seg",
        project_name="yolo_seg_finetuned",
    )
