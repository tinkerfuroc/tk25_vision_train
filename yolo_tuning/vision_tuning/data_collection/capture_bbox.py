"""RealSense bounding-box collection helpers."""

from yolo_tuning.create_dataset import RealSenseDatasetCreator
from yolo_tuning.vision_tuning.config import VisionConfig


class BBoxCollector:
    """Thin wrapper around the legacy RealSenseDatasetCreator for clarity."""

    def __init__(self, config: VisionConfig, output_dir: str | None = None):
        self.config = config
        self.output_dir = output_dir or config.dataset_dir

    def run(self) -> None:
        collector = RealSenseDatasetCreator(
            output_dir=self.output_dir,
            ontology_path=self.config.ontology_path,
            device=self.config.device,
        )
        collector.run()


def run_bbox_collection(config: VisionConfig, output_dir: str | None = None) -> None:
    BBoxCollector(config, output_dir).run()


__all__ = ["BBoxCollector", "run_bbox_collection"]
