from __future__ import annotations

import asyncio
import time
from typing import Iterator

import numpy as np
import pytest

from tk_vision.capture.live_camera import LiveCamera
from tk_vision.capture.source import Frame


class _FakeDriver:
    """A drop-in for RealSenseLiveSource that emits N synthetic frames."""

    def __init__(
        self,
        *,
        n: int = 60,
        width: int = 8,
        height: int = 6,
        fps: float = 60.0,
        frame_period_s: float = 0.005,
    ) -> None:
        self._n = n
        self.width = width
        self.height = height
        self.fps = fps
        self._period = frame_period_s

    def __enter__(self) -> "_FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    def frames(self) -> Iterator[Frame]:
        for i in range(self._n):
            arr = np.full((self.height, self.width, 3), i % 255, dtype=np.uint8)
            arr[0, 0] = (i, i, i)
            yield Frame(index=i, image_bgr=arr, timestamp_s=i / self.fps)
            time.sleep(self._period)


async def _drain(sub, until_idx: int, timeout: float) -> list[int]:
    seen: list[int] = []
    deadline = time.time() + timeout
    while time.time() < deadline and (not seen or seen[-1] < until_idx):
        try:
            f = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        seen.append(f.index)
    return seen


@pytest.mark.asyncio
async def test_lossless_subscriber_sees_every_frame() -> None:
    n = 30
    cam = LiveCamera(
        fps=60,
        resolution=(8, 6),
        driver_factory=lambda: _FakeDriver(n=n, frame_period_s=0.003),
    )
    sub = await cam.attach(lossless=True, name="rec")
    seen = await _drain(sub, until_idx=n - 1, timeout=5.0)
    await cam.detach(sub)
    assert seen == list(range(n)), f"lossless subscriber dropped frames: {seen}"


@pytest.mark.asyncio
async def test_two_subscribers_coexist() -> None:
    n = 30
    cam = LiveCamera(
        fps=60,
        resolution=(8, 6),
        driver_factory=lambda: _FakeDriver(n=n, frame_period_s=0.003),
    )
    rec = await cam.attach(lossless=True, name="rec")
    prev = await cam.attach(lossless=False, name="preview", queue_size=1)
    rec_seen, prev_seen = await asyncio.gather(
        _drain(rec, until_idx=n - 1, timeout=5.0),
        _drain(prev, until_idx=n - 1, timeout=5.0),
    )
    await cam.detach(rec)
    await cam.detach(prev)

    assert rec_seen == list(range(n)), "lossless subscriber lost frames while preview was attached"
    assert prev_seen == sorted(prev_seen)
    assert prev_seen[-1] == n - 1, "lossy preview should still see the last frame"
    # Lossy may drop intermediate frames, but it should be at most a small subset.
    assert len(prev_seen) >= 1


@pytest.mark.asyncio
async def test_pipeline_starts_and_stops_with_consumers() -> None:
    cam = LiveCamera(
        fps=60,
        resolution=(8, 6),
        driver_factory=lambda: _FakeDriver(n=10_000, frame_period_s=0.005),
    )
    assert cam.stats()["running"] is False
    s1 = await cam.attach(lossless=False)
    assert cam.stats()["running"] is True
    s2 = await cam.attach(lossless=False)
    assert cam.stats()["subscribers"] == 2
    await cam.detach(s1)
    assert cam.stats()["running"] is True
    await cam.detach(s2)
    # detach is asynchronous; wait briefly for the thread to join.
    for _ in range(20):
        if cam.stats()["running"] is False:
            break
        await asyncio.sleep(0.05)
    assert cam.stats()["running"] is False, "pipeline should stop once last subscriber detaches"


@pytest.mark.asyncio
async def test_attach_during_recording_does_not_disturb_recorder() -> None:
    n = 40
    cam = LiveCamera(
        fps=60,
        resolution=(8, 6),
        driver_factory=lambda: _FakeDriver(n=n, frame_period_s=0.005),
    )
    rec = await cam.attach(lossless=True, name="rec")

    # Drain 10 frames before preview joins.
    initial = []
    while len(initial) < 10:
        f = await asyncio.wait_for(rec.queue.get(), timeout=2.0)
        initial.append(f.index)
    assert initial == list(range(10))

    prev = await cam.attach(lossless=False, name="preview")

    rest_rec, _ = await asyncio.gather(
        _drain(rec, until_idx=n - 1, timeout=5.0),
        _drain(prev, until_idx=n - 1, timeout=5.0),
    )
    await cam.detach(rec)
    await cam.detach(prev)

    full_rec = initial + rest_rec
    assert full_rec == list(range(n)), (
        "recorder dropped or duplicated frames after preview attached"
    )
