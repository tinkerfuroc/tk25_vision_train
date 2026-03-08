"""Data collection package.

Keep this module lightweight to avoid importing hardware-only dependencies at package import time.
"""

from .splitter import split_yolo_dataset

__all__ = [
    "split_yolo_dataset",
]
