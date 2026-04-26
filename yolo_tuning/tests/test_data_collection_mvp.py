import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import supervision as sv

from yolo_tuning.vision_tuning.config import VisionConfig
from yolo_tuning.vision_tuning.commands.parser import build_parser
from yolo_tuning.vision_tuning.data_collection.crop_augment import CropAugmentConfig, build_crop_variants
from yolo_tuning.vision_tuning.data_collection.input_sources import iter_image_folder_frames, iter_video_frames
from yolo_tuning.vision_tuning.data_collection.seg_engine import SegEngineOptions, SegmentationCollectionEngine
from yolo_tuning.vision_tuning.data_collection.sam3_backend import SegmentationBatch
from yolo_tuning.vision_tuning.data_collection.web_review import WebReviewCollector


class _FakeBackend:
    def __init__(self, *_args, **_kwargs):
        pass

    def segment(self, image_bgr, prompts):
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        box = np.array([[w // 4, h // 4, 3 * w // 4, 3 * h // 4]], dtype=float)
        det = sv.Detections(xyxy=box, mask=np.array([mask]), confidence=np.array([0.9]), class_id=np.array([0]))
        return SegmentationBatch(detections=det, metadata={"phrases": ["object"]})

    def segment_with_points(self, image_bgr, points_xy):
        return np.zeros(image_bgr.shape[:2], dtype=bool)

    def segment_with_box(self, image_bgr, box_xyxy):
        return np.zeros(image_bgr.shape[:2], dtype=bool)


class TestDataCollectionMVP(unittest.TestCase):
    def test_cli_parser_contains_structured_commands(self):
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "create-seg",
                "--dataset-dir",
                "dataset_seg",
                "--input-mode",
                "images",
                "--source-path",
                "/tmp/images",
            ]
        )
        self.assertEqual(parsed.command, "create-seg")
        self.assertEqual(parsed.input_mode, "images")

    def test_iter_image_folder_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmp, "b.jpg"), img)
            cv2.imwrite(os.path.join(tmp, "a.png"), img)
            frames = list(iter_image_folder_frames(tmp))
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].shape, (32, 32, 3))

    def test_iter_video_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "sample.avi")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            frames = list(iter_video_frames(video_path))
            self.assertGreaterEqual(len(frames), 1)

    def test_crop_augment_variants(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=bool)
        mask[20:60, 20:60] = True
        cfg = CropAugmentConfig(enabled=True, variants=3, scale_min=1.05, scale_max=1.2, min_iou=0.95)
        crops = build_crop_variants(image, [mask], cfg)
        self.assertGreaterEqual(len(crops), 1)
        self.assertTrue(all(c.shape[0] > 0 and c.shape[1] > 0 for c in crops))

    def test_web_review_delete_last_detection_clears_pending_frame(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        det = sv.Detections(
            xyxy=np.array([[8, 8, 24, 24]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        review = WebReviewCollector(["object"], lambda *_args: None, mode="bbox", live_frame_provider=lambda: None)

        self.assertTrue(review.update_review(frame, det, det.class_id))
        ok, message = review._handle_action("delete")

        self.assertTrue(ok)
        self.assertEqual(message, "Deleted all detections")
        self.assertFalse(review.has_pending_review())

    def test_web_review_resizes_segmentation_masks_for_display(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        det = sv.Detections(
            xyxy=np.array([[8, 8, 24, 24]], dtype=float),
            mask=np.array([mask]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        review = WebReviewCollector(["object"], lambda *_args: None, mode="seg", live_frame_provider=lambda: None)

        annotated = review._annotate_segmentation(frame, det, ["object 0.90"], selected_idx=0)

        self.assertGreater(int(annotated.sum()), 0)

    @mock.patch("yolo_tuning.vision_tuning.data_collection.seg_engine.OfficialSAM3Backend", _FakeBackend)
    def test_segmentation_engine_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = os.path.join(tmp, "source")
            output_dir = os.path.join(tmp, "output")
            os.makedirs(source_dir, exist_ok=True)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(source_dir, "000.jpg"), image)

            ontology_path = os.path.join(tmp, "ontology.json")
            with open(ontology_path, "w") as f:
                json.dump({"object": "Object"}, f)

            cfg = VisionConfig().override(
                dataset_dir=tmp,
                seg_dataset_dir=tmp,
                ontology_path=ontology_path,
                checkpoint_dir=tmp,
                device="cpu",
            )
            opts = SegEngineOptions(input_mode="images", source_path=source_dir, max_frames=1)
            engine = SegmentationCollectionEngine(cfg, output_dir, opts)
            saved = engine.run()
            self.assertGreaterEqual(saved, 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "images", "000000.jpg")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "labels", "000000.txt")))


if __name__ == "__main__":
    unittest.main()
