"""Backward-compatible wrapper for the streaming segmentation collector."""

from dotenv import load_dotenv

from yolo_tuning.create_dataset_seg_stream import RealSenseSegStreamDatasetCreator, pre_cache_tokenizer


class RealSenseVideoDatasetCreator(RealSenseSegStreamDatasetCreator):
    """Alias to keep old imports working."""


def main():
    load_dotenv()
    pre_cache_tokenizer()
    RealSenseVideoDatasetCreator().run()


if __name__ == "__main__":
    main()
