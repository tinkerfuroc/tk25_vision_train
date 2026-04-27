"""Frame-range chunker for video propagation.

Splits a [start, end) range into chunks of `size` with `overlap` frames at the
start of each chunk that re-process the tail of the previous chunk. This gives
the propagator a smoother handoff and lets us recover cleanly on chunk OOM by
halving the size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A single propagation chunk.

    `start` and `end` are absolute clip frame indices. `core_start` is the
    first frame whose output is treated as authoritative (frames in
    [start, core_start) are written as ``source='propagated_overlap'``
    and skipped on YOLO export to avoid duplicates).
    """

    start: int
    core_start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start


def chunk_frames(*, start: int, end: int, size: int, overlap: int) -> list[Chunk]:
    """Split [start, end) into chunks.

    Args:
        start: first frame to propagate (inclusive)
        end:   one past last frame (exclusive)
        size:  preferred chunk size (e.g. 50)
        overlap: number of frames at chunk start re-processed for handoff
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if end <= start:
        return []

    chunks: list[Chunk] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + size, end)
        chunk_start_with_overlap = max(start, cursor - overlap) if chunks else cursor
        chunks.append(Chunk(start=chunk_start_with_overlap, core_start=cursor, end=chunk_end))
        cursor = chunk_end
    return chunks
