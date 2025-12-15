"""Utilities for building datasets and fine-tuning YOLO detectors/segmenters."""

__all__ = [
    "split_yolo_dataset",
    "merge_yolo_dataset",
    "BBoxCollector",
    "SegmentationCollector",
    "SegmentationStreamCollector",
    "SAM3SegmentationCollector",
    "SAM3Mask",
    "run_bbox_collection",
    "run_seg_collection",
    "run_seg_stream_collection",
    "train_detector",
    "train_segmenter",
    "run_live_detection",
    "run_live_segmentation",
    "VisionConfig",
    "Ontology",
]


def __getattr__(name):
    if name == "VisionConfig":
        from .config import VisionConfig

        return VisionConfig
    if name == "Ontology":
        from .ontology import Ontology

        return Ontology
    if name == "split_yolo_dataset":
        from .data_collection.splitter import split_yolo_dataset

        return split_yolo_dataset
    if name == "merge_yolo_dataset":
        from .data_collection.merger import merge_yolo_dataset

        return merge_yolo_dataset
    if name == "BBoxCollector" or name == "run_bbox_collection":
        from .data_collection.capture_bbox import BBoxCollector, run_bbox_collection

        return {"BBoxCollector": BBoxCollector, "run_bbox_collection": run_bbox_collection}[name]
    if name == "SegmentationCollector" or name == "run_seg_collection":
        from .data_collection.capture_seg import SegmentationCollector, run_seg_collection

        return {"SegmentationCollector": SegmentationCollector, "run_seg_collection": run_seg_collection}[name]
    if name == "SegmentationStreamCollector" or name == "run_seg_stream_collection":
        from .data_collection.capture_seg_stream import SegmentationStreamCollector, run_seg_stream_collection

        return {
            "SegmentationStreamCollector": SegmentationStreamCollector,
            "run_seg_stream_collection": run_seg_stream_collection,
        }[name]
    if name == "SAM3SegmentationCollector" or name == "SAM3Mask":
        from .data_collection.sam3 import SAM3Mask, SAM3SegmentationCollector

        return {"SAM3SegmentationCollector": SAM3SegmentationCollector, "SAM3Mask": SAM3Mask}[name]
    if name == "train_detector" or name == "train_segmenter":
        from .training import train_detector, train_segmenter

        return {"train_detector": train_detector, "train_segmenter": train_segmenter}[name]
    if name == "run_live_detection" or name == "run_live_segmentation":
        from .testing import run_live_detection, run_live_segmentation

        return {"run_live_detection": run_live_detection, "run_live_segmentation": run_live_segmentation}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
