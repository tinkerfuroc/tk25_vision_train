"""RealSense stream-based segmentation collection."""

from yolo_tuning.create_dataset_seg_stream import RealSenseSegStreamDatasetCreator
from yolo_tuning.vision_tuning.config import VisionConfig


class SegmentationStreamCollector:
    """Wrapper for the continuous RealSense stream segmentation workflow."""

    def __init__(self, config: VisionConfig, output_dir: str | None = None):
        self.config = config
        self.output_dir = output_dir or config.seg_dataset_dir

    def run(self) -> None:
        collector = RealSenseSegStreamDatasetCreator(
            output_dir=self.output_dir,
            ontology_path=self.config.ontology_path,
            device=self.config.device,
        )
        collector.run()


def run_seg_stream_collection(config: VisionConfig, output_dir: str | None = None) -> None:
    SegmentationStreamCollector(config, output_dir).run()


__all__ = ["SegmentationStreamCollector", "run_seg_stream_collection"]
