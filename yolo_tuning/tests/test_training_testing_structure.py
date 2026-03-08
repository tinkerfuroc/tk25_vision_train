import unittest
from unittest import mock

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.testing.live import run_live_detection, run_live_segmentation
from yolo_tuning.vision_tuning.testing.workflow import run_detection_live
from yolo_tuning.vision_tuning.training.yolo import train_detector, train_segmenter
from yolo_tuning.vision_tuning.training.workflow import run_detector_training, run_segmentation_training


class TestTrainingTestingStructure(unittest.TestCase):
    @mock.patch("yolo_tuning.vision_tuning.training.workflow.run_training_job")
    def test_run_detector_training_uses_detection_weights(self, mock_run_training_job):
        mock_run_training_job.return_value = ("best.pt", "/tmp/runs")
        cfg = VisionConfig().override(dataset_dir="dataset_det", seg_dataset_dir="dataset_seg", checkpoint_dir="/tmp/runs")
        run_detector_training(cfg, dataset_path="dataset_det", epochs=2, batch=1, imgsz=320, train_ratio=0.9)
        kwargs = mock_run_training_job.call_args.kwargs
        self.assertEqual(kwargs["weights"], cfg.detection_weights)
        self.assertEqual(kwargs["job"].project_name, "yolo_finetuned")

    @mock.patch("yolo_tuning.vision_tuning.training.workflow.run_training_job")
    def test_run_seg_training_uses_seg_weights(self, mock_run_training_job):
        mock_run_training_job.return_value = ("best.pt", "/tmp/runs")
        cfg = VisionConfig().override(dataset_dir="dataset_det", seg_dataset_dir="dataset_seg", checkpoint_dir="/tmp/runs")
        run_segmentation_training(cfg, dataset_path="dataset_seg", epochs=2, batch=1, imgsz=320, train_ratio=0.9)
        kwargs = mock_run_training_job.call_args.kwargs
        self.assertEqual(kwargs["weights"], cfg.segmentation_weights)
        self.assertEqual(kwargs["job"].project_name, "yolo_seg_finetuned")

    @mock.patch("yolo_tuning.vision_tuning.testing.workflow.os.path.exists", return_value=True)
    @mock.patch("yolo_tuning.vision_tuning.testing.workflow.run_realsense_live_test")
    def test_run_detection_live_invokes_core_runner(self, mock_core, _mock_exists):
        run_detection_live("model.pt")
        mock_core.assert_called_once()
        kwargs = mock_core.call_args.kwargs
        self.assertEqual(kwargs["task"], "bbox")
        self.assertEqual(kwargs["width"], 640)
        self.assertEqual(kwargs["height"], 480)

    @mock.patch("yolo_tuning.vision_tuning.training.yolo.run_detector_training")
    def test_train_detector_wrapper_delegates(self, mock_run):
        mock_run.return_value = ("best.pt", "runs")
        cfg = VisionConfig()
        train_detector(cfg, dataset_path="dataset", epochs=3, batch=2, imgsz=320, train_ratio=0.7)
        mock_run.assert_called_once()

    @mock.patch("yolo_tuning.vision_tuning.training.yolo.run_segmentation_training")
    def test_train_segmenter_wrapper_delegates(self, mock_run):
        mock_run.return_value = ("best.pt", "runs")
        cfg = VisionConfig()
        train_segmenter(cfg, dataset_path="dataset_seg", epochs=3, batch=2, imgsz=320, train_ratio=0.7)
        mock_run.assert_called_once()

    @mock.patch("yolo_tuning.vision_tuning.testing.live.run_detection_live")
    def test_live_detection_wrapper_delegates(self, mock_run):
        run_live_detection("weights.pt")
        mock_run.assert_called_once_with(model_path="weights.pt")

    @mock.patch("yolo_tuning.vision_tuning.testing.live.run_segmentation_live")
    def test_live_segmentation_wrapper_delegates(self, mock_run):
        run_live_segmentation("weights_seg.pt")
        mock_run.assert_called_once_with(model_path="weights_seg.pt")


if __name__ == "__main__":
    unittest.main()
