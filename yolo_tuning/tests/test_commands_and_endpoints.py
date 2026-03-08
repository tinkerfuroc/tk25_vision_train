import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from yolo_tuning.vision_tuning.cli import main as cli_main
from yolo_tuning.vision_tuning.commands.runner import run_cli
from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.endpoints.collect import split_dataset


class TestCommandsAndEndpoints(unittest.TestCase):
    @mock.patch("yolo_tuning.vision_tuning.commands.runner.collect_seg")
    def test_run_cli_dispatch_create_seg(self, mock_collect_seg):
        run_cli(
            [
                "create-seg",
                "--dataset-dir",
                "dataset_seg",
                "--input-mode",
                "images",
                "--source-path",
                "/tmp/images",
                "--enable-crop-augment",
                "--crop-variants",
                "3",
            ]
        )
        self.assertEqual(mock_collect_seg.call_count, 1)
        kwargs = mock_collect_seg.call_args.kwargs
        self.assertEqual(kwargs["dataset_dir"], "dataset_seg")
        self.assertEqual(kwargs["input_mode"], "images")
        self.assertEqual(kwargs["source_path"], "/tmp/images")
        self.assertTrue(kwargs["enable_crop_augment"])
        self.assertEqual(kwargs["crop_variants"], 3)

    @mock.patch("yolo_tuning.vision_tuning.commands.runner.train_bbox")
    def test_run_cli_dispatch_train_bbox_and_prints_paths(self, mock_train_bbox):
        mock_train_bbox.return_value = ("/tmp/yolo_finetuned_best.pt", "/tmp/runs/yolo_finetuned")
        output = io.StringIO()
        with redirect_stdout(output):
            run_cli(["train-bbox", "--dataset-dir", "dataset_det", "--epochs", "1"])
        text = output.getvalue()
        self.assertIn("Training complete. Results: /tmp/runs/yolo_finetuned", text)
        self.assertIn("Best model copied to: /tmp/yolo_finetuned_best.pt", text)
        self.assertEqual(mock_train_bbox.call_count, 1)

    @mock.patch("yolo_tuning.vision_tuning.commands.runner.train_seg")
    def test_run_cli_dispatch_train_seg_and_prints_paths(self, mock_train_seg):
        mock_train_seg.return_value = ("/tmp/yolo_seg_finetuned_best.pt", "/tmp/runs/yolo_seg_finetuned")
        output = io.StringIO()
        with redirect_stdout(output):
            run_cli(["train-seg", "--dataset-dir", "dataset_seg", "--epochs", "1"])
        text = output.getvalue()
        self.assertIn("Training complete. Results: /tmp/runs/yolo_seg_finetuned", text)
        self.assertIn("Best model copied to: /tmp/yolo_seg_finetuned_best.pt", text)
        self.assertEqual(mock_train_seg.call_count, 1)

    @mock.patch("yolo_tuning.vision_tuning.commands.runner.test_bbox")
    def test_run_cli_dispatch_test_bbox(self, mock_test_bbox):
        run_cli(["test-bbox", "--model-path", "weights.pt"])
        mock_test_bbox.assert_called_once_with("weights.pt")

    @mock.patch("yolo_tuning.vision_tuning.commands.runner.collect_seg_stream")
    def test_run_cli_dispatch_create_seg_stream(self, mock_collect_seg_stream):
        run_cli(
            [
                "create-seg-stream",
                "--dataset-dir",
                "dataset_seg",
                "--input-mode",
                "video",
                "--source-path",
                "/tmp/demo.mp4",
            ]
        )
        self.assertEqual(mock_collect_seg_stream.call_count, 1)
        kwargs = mock_collect_seg_stream.call_args.kwargs
        self.assertEqual(kwargs["dataset_dir"], "dataset_seg")
        self.assertEqual(kwargs["input_mode"], "video")
        self.assertEqual(kwargs["source_path"], "/tmp/demo.mp4")

    @mock.patch("yolo_tuning.vision_tuning.endpoints.collect.split_yolo_dataset")
    def test_split_endpoint_uses_config_seed(self, mock_split):
        cfg = VisionConfig().override(dataset_dir="dataset_det", seg_dataset_dir="dataset_seg")
        split_dataset(cfg, dataset_dir="dataset_det", train_ratio=0.75, seed=None)
        mock_split.assert_called_once_with("dataset_det", train_ratio=0.75, seed=cfg.seed)

    @mock.patch("yolo_tuning.vision_tuning.cli.run_cli")
    def test_cli_main_is_thin_wrapper(self, mock_run_cli):
        cli_main(["split", "--dataset-dir", "dataset_det"])
        mock_run_cli.assert_called_once_with(["split", "--dataset-dir", "dataset_det"])


if __name__ == "__main__":
    unittest.main()
