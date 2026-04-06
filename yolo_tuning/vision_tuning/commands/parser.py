import argparse


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda:0). Defaults to auto-detect.")
    parser.add_argument("--ontology-path", dest="ontology_path", default=None, help="Path to ontology JSON.")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None, help="Where to save YOLO runs.")


def _add_seg_collection_args(parser: argparse.ArgumentParser, *, default_max_frames):
    parser.add_argument("--dataset-dir", default=None, help="Output dataset directory for segmentation.")
    parser.add_argument("--input-mode", choices=["realsense", "images", "video"], default="realsense")
    parser.add_argument("--source-path", default=None, help="Source path for images/video mode.")
    parser.add_argument("--enable-crop-augment", action="store_true", help="Enable crop diversity augmentation.")
    parser.add_argument("--crop-variants", type=int, default=2)
    parser.add_argument("--crop-scale-min", type=float, default=1.05)
    parser.add_argument("--crop-scale-max", type=float, default=1.30)
    parser.add_argument("--max-frames", type=int, default=default_max_frames)
    parser.add_argument("--no-review", action="store_true", help="Disable GUI review and save automatically.")
    parser.add_argument("--no-live-preview", action="store_true", help="Disable real-time visualization during collection.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision tuning utilities (dataset + training).")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common)

    bbox = sub.add_parser("create-bbox", parents=[common], help="Collect bounding box data from RealSense.")
    bbox.add_argument("--dataset-dir", default=None, help="Output dataset directory (images/ + labels/).")

    seg = sub.add_parser("create-seg", parents=[common], help="Collect segmentation masks.")
    _add_seg_collection_args(seg, default_max_frames=None)

    seg_stream = sub.add_parser("create-seg-stream", parents=[common], help="Collect segmentation masks from stream.")
    _add_seg_collection_args(seg_stream, default_max_frames=300)

    split = sub.add_parser("split", parents=[common], help="Split a raw dataset into train/val.")
    split.add_argument("--dataset-dir", default=None, help="Dataset root containing images/ and labels/.")
    split.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images for training.")
    split.add_argument("--seed", type=int, default=None, help="RNG seed (defaults to config).")

    train_bbox = sub.add_parser("train-bbox", parents=[common], help="Fine-tune YOLO detector.")
    train_bbox.add_argument("--dataset-dir", default=None, help="Dataset root (bounding boxes).")
    train_bbox.add_argument("--epochs", type=int, default=50)
    train_bbox.add_argument("--batch", type=int, default=8)
    train_bbox.add_argument("--imgsz", type=int, default=640)
    train_bbox.add_argument("--train-ratio", type=float, default=0.8, help="Split ratio if dataset not already split.")

    train_seg = sub.add_parser("train-seg", parents=[common], help="Fine-tune YOLO-seg.")
    train_seg.add_argument("--dataset-dir", default=None, help="Dataset root (segmentation).")
    train_seg.add_argument("--epochs", type=int, default=250)
    train_seg.add_argument("--batch", type=int, default=4)
    train_seg.add_argument("--imgsz", type=int, default=640)
    train_seg.add_argument("--train-ratio", type=float, default=0.8, help="Split ratio if dataset not already split.")

    test_bbox = sub.add_parser("test-bbox", parents=[common], help="Run live test for detector.")
    test_bbox.add_argument("--model-path", default=None, help="Path to trained detector weights.")

    test_seg = sub.add_parser("test-seg", parents=[common], help="Run live test for segmenter.")
    test_seg.add_argument("--model-path", default=None, help="Path to trained segmentation weights.")

    return parser
