import numpy as np

from yolo_tuning.vision_tuning.data_collection.sam3 import SAM3Mask, SAM3SegmentationCollector


class FakeSegmenter:
    def __init__(self):
        self.calls = 0

    def segment(self, frame, prompts):
        self.calls += 1
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[1:4, 1:4] = 1
        return [SAM3Mask(mask=mask, label=prompts[0] if prompts else "object")]


class FakeTracker:
    def __init__(self):
        self.calls = 0

    def track(self, prev_masks, frame):
        self.calls += 1
        shifted = np.roll(prev_masks[0].mask, 1, axis=1)
        return [SAM3Mask(mask=shifted, label=prev_masks[0].label)]


def test_sam3_collector_tracks_and_writes_labels(tmp_path):
    segmenter = FakeSegmenter()
    tracker = FakeTracker()
    collector = SAM3SegmentationCollector(
        output_dir=tmp_path.as_posix(),
        segmenter=segmenter,
        tracker=tracker,
        label_map={"object": 0},
    )

    frames = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(2)]
    saved = collector.collect(frames, prompts=["object"])

    assert saved == 2
    assert segmenter.calls >= 1
    assert tracker.calls == 1

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    assert len(list(images_dir.glob("*.jpg"))) == 2
    labels = list(labels_dir.glob("*.txt"))
    assert len(labels) == 2

    first_label = labels_dir / "sam3_00000.txt"
    content = first_label.read_text().strip().split()
    assert content and content[0] == "0"
