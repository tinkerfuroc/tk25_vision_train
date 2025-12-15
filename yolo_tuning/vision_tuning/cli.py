import argparse
from typing import Optional

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.merger import merge_yolo_dataset
from yolo_tuning.vision_tuning.data_collection.splitter import split_yolo_dataset
from yolo_tuning.vision_tuning.training import train_detector, train_segmenter


def _config_from_args(
    base: VisionConfig,
    *,
    dataset_dir: Optional[str],
    seg_dataset_dir: Optional[str],
    ontology: Optional[str],
    checkpoint: Optional[str],
    device: Optional[str],
) -> VisionConfig:
    return base.override(
        dataset_dir=dataset_dir,
        seg_dataset_dir=seg_dataset_dir,
        ontology_path=ontology,
        checkpoint_dir=checkpoint,
        device=device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision tuning utilities (dataset + training).")
    sub = parser.add_subparsers(dest="command", required=True)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--device", default=None, help="Force device (cpu/cuda:0). Defaults to auto-detect.")
    common_args.add_argument("--ontology-path", dest="ontology_path", default=None, help="Path to ontology JSON.")
    common_args.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None, help="Where to save YOLO runs.")

    bbox = sub.add_parser("create-bbox", parents=[common_args], help="Collect bounding box data from RealSense.")
    bbox.add_argument("--dataset-dir", default=None, help="Output dataset directory (images/ + labels/).")

    seg = sub.add_parser("create-seg", parents=[common_args], help="Collect segmentation masks from RealSense.")
    seg.add_argument("--dataset-dir", default=None, help="Output dataset directory for segmentation.")

    seg_stream = sub.add_parser("create-seg-stream", parents=[common_args], help="Collect segmentation masks from live stream.")
    seg_stream.add_argument("--dataset-dir", default=None, help="Output dataset directory for segmentation.")

    split = sub.add_parser("split", parents=[common_args], help="Split a raw dataset into train/val.")
    split.add_argument("--dataset-dir", default=None, help="Dataset root containing images/ and labels/.")
    split.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images for training.")
    split.add_argument("--seed", type=int, default=None, help="RNG seed (defaults to config).")

    merge = sub.add_parser("merge", parents=[common_args], help="Merge train/val back into flat YOLO dataset.")
    merge.add_argument("--dataset-dir", default=None, help="Dataset root containing split images/ and labels/.")

    sam3 = sub.add_parser("create-seg-sam3", parents=[common_args], help="Collect segmentation masks using SAM3.")
    sam3.add_argument("--images-dir", required=True, help="Directory of frames to segment/track.")
    sam3.add_argument("--dataset-dir", default=None, help="Output dataset directory for segmentation.")
    sam3.add_argument("--prompts", nargs="*", default=None, help="Prompts to guide language-grounded SAM3.")

    train_bbox = sub.add_parser("train-bbox", parents=[common_args], help="Fine-tune YOLO detector.")
    train_bbox.add_argument("--dataset-dir", default=None, help="Dataset root (bounding boxes).")
    train_bbox.add_argument("--epochs", type=int, default=50)
    train_bbox.add_argument("--batch", type=int, default=8)
    train_bbox.add_argument("--imgsz", type=int, default=640)
    train_bbox.add_argument("--train-ratio", type=float, default=0.8, help="Split ratio if dataset not already split.")

    train_seg = sub.add_parser("train-seg", parents=[common_args], help="Fine-tune YOLO-seg.")
    train_seg.add_argument("--dataset-dir", default=None, help="Dataset root (segmentation).")
    train_seg.add_argument("--epochs", type=int, default=250)
    train_seg.add_argument("--batch", type=int, default=4)
    train_seg.add_argument("--imgsz", type=int, default=640)
    train_seg.add_argument("--train-ratio", type=float, default=0.8, help="Split ratio if dataset not already split.")

    test_bbox = sub.add_parser("test-bbox", parents=[common_args], help="Run live test for detector.")
    test_bbox.add_argument("--model-path", default=None, help="Path to trained detector weights.")

    test_seg = sub.add_parser("test-seg", parents=[common_args], help="Run live test for segmenter.")
    test_seg.add_argument("--model-path", default=None, help="Path to trained segmentation weights.")

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_config = VisionConfig()
    if args.command == "create-bbox":
        from yolo_tuning.vision_tuning.data_collection.collectors import launch_bbox_collection

        cfg = _config_from_args(
            base_config,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=None,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        launch_bbox_collection(cfg, output_dir=args.dataset_dir)

    elif args.command == "create-seg":
        from yolo_tuning.vision_tuning.data_collection.collectors import launch_seg_collection

        cfg = _config_from_args(
            base_config,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        launch_seg_collection(cfg, output_dir=args.dataset_dir)

    elif args.command == "create-seg-stream":
        from yolo_tuning.vision_tuning.data_collection.collectors import launch_seg_stream_collection

        cfg = _config_from_args(
            base_config,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        launch_seg_stream_collection(cfg, output_dir=args.dataset_dir)

    elif args.command == "split":
        cfg = _config_from_args(
            base_config,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        dataset_root = args.dataset_dir or cfg.dataset_dir
        split_yolo_dataset(dataset_root, train_ratio=args.train_ratio, seed=args.seed or cfg.seed)

    elif args.command == "merge":
        cfg = _config_from_args(
            base_config,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        dataset_root = args.dataset_dir or cfg.dataset_dir
        images_moved, labels_moved = merge_yolo_dataset(dataset_root)
        print(f"Merged {images_moved} images and {labels_moved} labels into flat dataset at {dataset_root}")

    elif args.command == "train-bbox":
        cfg = _config_from_args(
            base_config,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=None,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        best_model, results_dir = train_detector(
            cfg,
            dataset_path=args.dataset_dir,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            train_ratio=args.train_ratio,
        )
        print(f"Training complete. Results: {results_dir}")
        if best_model:
            print(f"Best model copied to: {best_model}")

    elif args.command == "train-seg":
        cfg = _config_from_args(
            base_config,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        best_model, results_dir = train_segmenter(
            cfg,
            dataset_path=args.dataset_dir,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            train_ratio=args.train_ratio,
        )
        print(f"Training complete. Results: {results_dir}")
        if best_model:
            print(f"Best model copied to: {best_model}")

    elif args.command == "test-bbox":
        from yolo_tuning.vision_tuning.testing import run_live_detection

        run_live_detection(args.model_path)

    elif args.command == "test-seg":
        from yolo_tuning.vision_tuning.testing import run_live_segmentation

        run_live_segmentation(args.model_path)

    elif args.command == "create-seg-sam3":
        from yolo_tuning.vision_tuning.data_collection.sam3 import SAM3SegmentationCollector

        cfg = _config_from_args(
            base_config,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        collector = SAM3SegmentationCollector(
            output_dir=args.dataset_dir or cfg.seg_dataset_dir,
            ontology_path=args.ontology_path or cfg.ontology_path,
        )
        frames = collector.load_frames_from_dir(args.images_dir)
        saved = collector.collect(frames, prompts=args.prompts)
        print(f"SAM3 segmentation complete. Saved {saved} frame(s) to {collector.output_dir}")


if __name__ == "__main__":
    main()
