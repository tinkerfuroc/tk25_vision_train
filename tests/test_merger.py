import os

from yolo_tuning.vision_tuning.data_collection.merger import merge_yolo_dataset
from yolo_tuning.vision_tuning.data_collection.splitter import split_yolo_dataset


def _create_stub_dataset(dataset_root: str, count: int = 4) -> None:
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for idx in range(count):
        with open(os.path.join(images_dir, f"img_{idx}.jpg"), "wb") as f:
            f.write(b"img")
        with open(os.path.join(labels_dir, f"img_{idx}.txt"), "w") as f:
            f.write("0 0.5 0.5 1 1\n")


def test_split_and_merge_roundtrip(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    _create_stub_dataset(dataset_root.as_posix(), count=4)

    train_count, val_count = split_yolo_dataset(dataset_root.as_posix(), train_ratio=0.5, seed=0)
    assert train_count + val_count == 4

    images_moved, labels_moved = merge_yolo_dataset(dataset_root.as_posix())
    assert images_moved == 4
    assert labels_moved == 4

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    assert sorted(p.name for p in images_root.iterdir()) == [
        "img_0.jpg",
        "img_1.jpg",
        "img_2.jpg",
        "img_3.jpg",
    ]
    assert sorted(p.name for p in labels_root.iterdir()) == [
        "img_0.txt",
        "img_1.txt",
        "img_2.txt",
        "img_3.txt",
    ]

    # train/val folders removed after merge
    assert not (images_root / "train").exists()
    assert not (images_root / "val").exists()
    assert not (labels_root / "train").exists()
    assert not (labels_root / "val").exists()
