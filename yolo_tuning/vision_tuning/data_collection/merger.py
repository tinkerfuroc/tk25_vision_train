import os
import shutil
from typing import Tuple


def merge_yolo_dataset(dataset_path: str) -> Tuple[int, int]:
    """Merge a split YOLO dataset back into flat images/ and labels/ folders.

    Moves files from images/train, images/val, labels/train, labels/val
    back into images/ and labels/. Returns a tuple of (images_moved, labels_moved).
    """
    images_root = os.path.join(dataset_path, "images")
    labels_root = os.path.join(dataset_path, "labels")
    train_images = os.path.join(images_root, "train")
    val_images = os.path.join(images_root, "val")
    train_labels = os.path.join(labels_root, "train")
    val_labels = os.path.join(labels_root, "val")

    if not all(os.path.isdir(p) for p in (images_root, labels_root)):
        raise FileNotFoundError("Expected images/ and labels/ folders inside the dataset directory.")

    split_dirs = (train_images, val_images, train_labels, val_labels)
    if not any(os.path.isdir(p) for p in split_dirs):
        raise FileNotFoundError("No split train/val folders found to merge.")

    os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    def _merge_folder(source_dir: str, dest_dir: str) -> int:
        if not os.path.isdir(source_dir):
            return 0
        moved = 0
        for name in os.listdir(source_dir):
            src = os.path.join(source_dir, name)
            if os.path.isdir(src):
                continue
            dest = os.path.join(dest_dir, name)
            if os.path.exists(dest):
                raise FileExistsError(f"Destination file already exists: {dest}")
            shutil.move(src, dest)
            moved += 1
        return moved

    images_moved = _merge_folder(train_images, images_root) + _merge_folder(val_images, images_root)
    labels_moved = _merge_folder(train_labels, labels_root) + _merge_folder(val_labels, labels_root)

    for maybe_empty in split_dirs:
        if os.path.isdir(maybe_empty) and not os.listdir(maybe_empty):
            os.rmdir(maybe_empty)

    return images_moved, labels_moved


__all__ = ["merge_yolo_dataset"]
