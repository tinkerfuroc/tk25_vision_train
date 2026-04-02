"""Stream segmentation collection helpers (SAM3 backend + input modes)."""

from typing import Optional

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.data_collection.seg_engine import SegEngineOptions, SegmentationCollectionEngine


class SegmentationStreamCollector:
    """Collector wrapper for stream-like segmentation workflows."""

    def __init__(
        self,
        config: VisionConfig,
        output_dir: str | None = None,
        *,
        input_mode: str = "realsense",
        source_path: Optional[str] = None,
        enable_crop_augment: bool = False,
        crop_variants: int = 2,
        crop_scale_min: float = 1.05,
        crop_scale_max: float = 1.30,
        max_frames: Optional[int] = 300,
        enable_review: bool = True,
        enable_live_preview: bool = True,
    ):
        self.config = config
        self.output_dir = output_dir or config.seg_dataset_dir
        self.options = SegEngineOptions(
            input_mode=input_mode,
            source_path=source_path,
            enable_crop_augment=enable_crop_augment,
            crop_variants=crop_variants,
            crop_scale_min=crop_scale_min,
            crop_scale_max=crop_scale_max,
            max_frames=max_frames,
            enable_review=enable_review,
            enable_live_preview=enable_live_preview,
        )

    def run(self) -> None:
        engine = SegmentationCollectionEngine(self.config, self.output_dir, self.options)
        saved = engine.run()
        print(f"Saved {saved} sample(s) to {self.output_dir}")


def run_seg_stream_collection(
    config: VisionConfig,
    output_dir: str | None = None,
    *,
    input_mode: str = "realsense",
    source_path: Optional[str] = None,
    enable_crop_augment: bool = False,
    crop_variants: int = 2,
    crop_scale_min: float = 1.05,
    crop_scale_max: float = 1.30,
    max_frames: Optional[int] = 300,
    enable_review: bool = True,
    enable_live_preview: bool = True,
) -> None:
    SegmentationStreamCollector(
        config,
        output_dir,
        input_mode=input_mode,
        source_path=source_path,
        enable_crop_augment=enable_crop_augment,
        crop_variants=crop_variants,
        crop_scale_min=crop_scale_min,
        crop_scale_max=crop_scale_max,
        max_frames=max_frames,
        enable_review=enable_review,
        enable_live_preview=enable_live_preview,
    ).run()


__all__ = ["SegmentationStreamCollector", "run_seg_stream_collection"]
