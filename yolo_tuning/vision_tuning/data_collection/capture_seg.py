"""RealSense segmentation-mask collection helpers."""

from yolo_tuning.create_dataset_seg import RealSenseDatasetCreator
from yolo_tuning.vision_tuning.config import VisionConfig


class SegmentationCollector:
    """Wrapper for LangSAM + SAM assisted mask collection."""

    def __init__(self, config: VisionConfig, output_dir: str | None = None):
        self.config = config
        self.output_dir = output_dir or config.seg_dataset_dir

    def run(self) -> None:
        collector = RealSenseDatasetCreator(
            output_dir=self.output_dir,
            ontology_path=self.config.ontology_path,
            device=self.config.device,
        )
        collector.run()


def run_seg_collection(config: VisionConfig, output_dir: str | None = None) -> None:
    SegmentationCollector(config, output_dir).run()


__all__ = ["SegmentationCollector", "run_seg_collection"]
