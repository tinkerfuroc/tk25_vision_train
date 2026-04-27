from __future__ import annotations

from tk_vision.annotate.chunker import Chunk, chunk_frames


def test_single_chunk_no_overlap_when_smaller_than_size() -> None:
    cs = chunk_frames(start=0, end=20, size=50, overlap=4)
    assert cs == [Chunk(start=0, core_start=0, end=20)]


def test_two_chunks_with_overlap() -> None:
    cs = chunk_frames(start=0, end=100, size=50, overlap=4)
    assert len(cs) == 2
    assert cs[0] == Chunk(start=0, core_start=0, end=50)
    assert cs[1] == Chunk(start=46, core_start=50, end=100)


def test_three_chunks() -> None:
    cs = chunk_frames(start=0, end=130, size=50, overlap=4)
    assert len(cs) == 3
    assert cs[0].start == 0 and cs[0].end == 50
    assert cs[1].start == 46 and cs[1].core_start == 50 and cs[1].end == 100
    assert cs[2].start == 96 and cs[2].core_start == 100 and cs[2].end == 130


def test_chunk_start_offset() -> None:
    cs = chunk_frames(start=10, end=110, size=50, overlap=4)
    assert cs[0] == Chunk(start=10, core_start=10, end=60)
    assert cs[1] == Chunk(start=56, core_start=60, end=110)


def test_overlap_clamps_at_start_boundary() -> None:
    """Overlap into chunk 1 should not reach below `start`."""
    cs = chunk_frames(start=5, end=120, size=50, overlap=10)
    assert cs[0].start == 5
    assert cs[1].start == 45  # max(5, 55-10)
    assert cs[1].core_start == 55


def test_empty_range() -> None:
    assert chunk_frames(start=0, end=0, size=50, overlap=4) == []
    assert chunk_frames(start=10, end=5, size=50, overlap=4) == []


def test_invalid_args() -> None:
    import pytest
    with pytest.raises(ValueError):
        chunk_frames(start=0, end=10, size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_frames(start=0, end=10, size=4, overlap=4)
