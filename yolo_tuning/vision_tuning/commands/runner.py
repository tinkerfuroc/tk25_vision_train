from typing import Optional

from yolo_tuning.vision_tuning.commands.parser import build_parser
from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.endpoints import (
    collect_bbox,
    collect_seg,
    collect_seg_stream,
    split_dataset,
    test_bbox,
    test_seg,
    train_bbox,
    train_seg,
)


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


def run_cli(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = VisionConfig()

    if args.command == "create-bbox":
        cfg = _config_from_args(
            base,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=None,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        collect_bbox(cfg, dataset_dir=args.dataset_dir)
        return

    if args.command == "create-seg":
        cfg = _config_from_args(
            base,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        collect_seg(
            cfg,
            dataset_dir=args.dataset_dir,
            input_mode=args.input_mode,
            source_path=args.source_path,
            enable_crop_augment=args.enable_crop_augment,
            crop_variants=args.crop_variants,
            crop_scale_min=args.crop_scale_min,
            crop_scale_max=args.crop_scale_max,
            max_frames=args.max_frames,
            enable_review=not args.no_review,
            enable_live_preview=not args.no_live_preview,
        )
        return

    if args.command == "create-seg-stream":
        cfg = _config_from_args(
            base,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        collect_seg_stream(
            cfg,
            dataset_dir=args.dataset_dir,
            input_mode=args.input_mode,
            source_path=args.source_path,
            enable_crop_augment=args.enable_crop_augment,
            crop_variants=args.crop_variants,
            crop_scale_min=args.crop_scale_min,
            crop_scale_max=args.crop_scale_max,
            max_frames=args.max_frames,
            enable_review=not args.no_review,
            enable_live_preview=not args.no_live_preview,
        )
        return

    if args.command == "split":
        cfg = _config_from_args(
            base,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        split_dataset(cfg, dataset_dir=args.dataset_dir, train_ratio=args.train_ratio, seed=args.seed)
        return

    if args.command == "train-bbox":
        cfg = _config_from_args(
            base,
            dataset_dir=args.dataset_dir,
            seg_dataset_dir=None,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        best_model, results_dir = train_bbox(
            cfg,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            train_ratio=args.train_ratio,
        )
        print(f"Training complete. Results: {results_dir}")
        if best_model:
            print(f"Best model copied to: {best_model}")
        return

    if args.command == "train-seg":
        cfg = _config_from_args(
            base,
            dataset_dir=None,
            seg_dataset_dir=args.dataset_dir,
            ontology=args.ontology_path,
            checkpoint=args.checkpoint_dir,
            device=args.device,
        )
        best_model, results_dir = train_seg(
            cfg,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            train_ratio=args.train_ratio,
        )
        print(f"Training complete. Results: {results_dir}")
        if best_model:
            print(f"Best model copied to: {best_model}")
        return

    if args.command == "test-bbox":
        test_bbox(args.model_path)
        return

    if args.command == "test-seg":
        test_seg(args.model_path)
        return
