__all__ = [
    "RealSenseBBoxCollector",
    "RealSenseSegCollector",
    "RealSenseSegStreamCollector",
    "BBoxCollector",
    "SegmentationCollector",
    "SegmentationStreamCollector",
    "run_bbox_collection",
    "run_seg_collection",
    "run_seg_stream_collection",
    "split_yolo_dataset",
    "merge_yolo_dataset",
    "SAM3Mask",
    "SAM3SegmentationCollector",
    "cache_bbox_tokenizer",
    "cache_seg_tokenizer",
    "cache_seg_stream_tokenizer",
]


def __getattr__(name):
    if name in ("RealSenseBBoxCollector", "cache_bbox_tokenizer"):
        from .bbox import RealSenseBBoxCollector, pre_cache_tokenizer as cache_bbox_tokenizer

        return {"RealSenseBBoxCollector": RealSenseBBoxCollector, "cache_bbox_tokenizer": cache_bbox_tokenizer}[name]
    if name in ("RealSenseSegCollector", "cache_seg_tokenizer"):
        from .seg import RealSenseSegCollector, pre_cache_tokenizer as cache_seg_tokenizer

        return {"RealSenseSegCollector": RealSenseSegCollector, "cache_seg_tokenizer": cache_seg_tokenizer}[name]
    if name in ("RealSenseSegStreamCollector", "cache_seg_stream_tokenizer"):
        from .seg_stream import RealSenseSegStreamCollector, pre_cache_tokenizer as cache_seg_stream_tokenizer

        return {
            "RealSenseSegStreamCollector": RealSenseSegStreamCollector,
            "cache_seg_stream_tokenizer": cache_seg_stream_tokenizer,
        }[name]
    if name in ("BBoxCollector", "run_bbox_collection"):
        from .capture_bbox import BBoxCollector, run_bbox_collection

        return {"BBoxCollector": BBoxCollector, "run_bbox_collection": run_bbox_collection}[name]
    if name in ("SegmentationCollector", "run_seg_collection"):
        from .capture_seg import SegmentationCollector, run_seg_collection

        return {"SegmentationCollector": SegmentationCollector, "run_seg_collection": run_seg_collection}[name]
    if name in ("SegmentationStreamCollector", "run_seg_stream_collection"):
        from .capture_seg_stream import SegmentationStreamCollector, run_seg_stream_collection

        return {
            "SegmentationStreamCollector": SegmentationStreamCollector,
            "run_seg_stream_collection": run_seg_stream_collection,
        }[name]
    if name in ("split_yolo_dataset",):
        from .splitter import split_yolo_dataset

        return split_yolo_dataset
    if name in ("merge_yolo_dataset",):
        from .merger import merge_yolo_dataset

        return merge_yolo_dataset
    if name in ("SAM3SegmentationCollector", "SAM3Mask"):
        from .sam3 import SAM3Mask, SAM3SegmentationCollector

        return {"SAM3SegmentationCollector": SAM3SegmentationCollector, "SAM3Mask": SAM3Mask}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
