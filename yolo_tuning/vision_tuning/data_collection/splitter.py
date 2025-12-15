import os
import random
import shutil
from typing import Iterable, Sequence, Tuple


def _list_image_stems(images_dir: str, extensions: Sequence[str]) -> Iterable[str]:
    for name in os.listdir(images_dir):
        if any(name.lower().endswith(ext) for ext in extensions):
            yield os.path.splitext(name)[0]


def split_yolo_dataset(dataset_path: str, train_ratio: float = 0.8, seed: int = 42) -> Tuple[int, int]:
    """Split a YOLO-format dataset into train/val folders.

    Returns:
        (train_count, val_count)
    """
    images_source_dir = os.path.join(dataset_path, "images")
    labels_source_dir = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_source_dir) or not os.path.isdir(labels_source_dir):
        raise FileNotFoundError("Expected images/ and labels/ folders inside the dataset directory.")

    rng = random.Random(seed)

    train_images_dest_dir = os.path.join(images_source_dir, "train")
    val_images_dest_dir = os.path.join(images_source_dir, "val")
    train_labels_dest_dir = os.path.join(labels_source_dir, "train")
    val_labels_dest_dir = os.path.join(labels_source_dir, "val")

    for dest in (train_images_dest_dir, val_images_dest_dir, train_labels_dest_dir, val_labels_dest_dir):
        os.makedirs(dest, exist_ok=True)

    image_stems = list(_list_image_stems(images_source_dir, (".jpg", ".jpeg", ".png")))
    if not image_stems:
        raise FileNotFoundError("No images found to split. Did you point at the correct dataset?")

    rng.shuffle(image_stems)
    split_index = int(len(image_stems) * train_ratio)
    train_files = image_stems[:split_index]
    val_files = image_stems[split_index:]

    def _move_group(files, img_dest, lbl_dest):
        for stem in files:
            for ext in (".jpg", ".jpeg", ".png"):
                src_img = os.path.join(images_source_dir, stem + ext)
                if os.path.exists(src_img):
                    break
            else:
                continue

            src_lbl = os.path.join(labels_source_dir, stem + ".txt")
            if not os.path.exists(src_lbl):
                continue

            shutil.move(src_img, os.path.join(img_dest, os.path.basename(src_img)))
            shutil.move(src_lbl, os.path.join(lbl_dest, os.path.basename(src_lbl)))

    _move_group(train_files, train_images_dest_dir, train_labels_dest_dir)
    _move_group(val_files, val_images_dest_dir, val_labels_dest_dir)

    return len(train_files), len(val_files)
