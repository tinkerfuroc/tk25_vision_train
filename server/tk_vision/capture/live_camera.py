from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Callable, ContextManager, Iterator, Optional

import numpy as np

from .source import Frame

log = logging.getLogger("tk_vision.live_camera")


@dataclass
class Subscriber:
    name: str
    queue: asyncio.Queue
    lossless: bool
    loop: asyncio.AbstractEventLoop


def _lossy_put(q: asyncio.Queue, frame: Frame) -> None:
    try:
        q.put_nowait(frame)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            pass


def _lossless_put(q: asyncio.Queue, frame: Frame, sub_name: str) -> None:
    try:
        q.put_nowait(frame)
    except asyncio.QueueFull:
        # Buffer overflow on a lossless subscriber means the consumer fell
        # too far behind. We can't block the capture thread from here; drop
        # oldest and warn so the issue is visible.
        log.warning("LiveCamera: lossless subscriber %s queue full; dropping oldest", sub_name)
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            pass


DriverFactory = Callable[[], "ContextManager[object]"]


class LiveCamera:
    """Single owner of the RealSense color stream, fans frames out to subscribers.

    The pipeline is opened on first attach and closed when the last subscriber
    detaches. A daemon Python thread reads frames and broadcasts each to every
    subscriber's `asyncio.Queue` via `loop.call_soon_threadsafe`.

    Lossy subscribers (preview): queue size 1, oldest dropped if backed up.
    Lossless subscribers (recording): queue size 256 (~8 s @ 30 fps); a warning
    fires if it ever fills, and the oldest is dropped to keep the camera fluid.

    The driver factory can be overridden for tests so this class is exercised
    without RealSense hardware.
    """

    DEFAULT_LOSSY_QUEUE_SIZE = 1
    DEFAULT_LOSSLESS_QUEUE_SIZE = 256
    PIPELINE_START_TIMEOUT_S = 8.0
    PIPELINE_STOP_TIMEOUT_S = 3.0

    def __init__(
        self,
        *,
        fps: int,
        resolution: tuple[int, int],
        driver_factory: Optional[DriverFactory] = None,
    ) -> None:
        self.fps = int(fps)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self._driver_factory = driver_factory
        self._subscribers: list[Subscriber] = []
        self._sub_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._driver_error: BaseException | None = None
        self._frames_seen = 0
        self._last_error: str | None = None
        self._observed_width: int | None = None
        self._observed_height: int | None = None
        self._observed_fps: float | None = None

    # ------------------------------------------------------------------ inspection
    def device_present(self) -> bool:
        if self._driver_factory is not None:
            return True
        try:
            import pyrealsense2 as rs  # noqa: WPS433

            return any(True for _ in rs.context().query_devices())
        except Exception:
            return False

    def stats(self) -> dict[str, object]:
        with self._sub_lock:
            n = len(self._subscribers)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "subscribers": n,
            "frames_seen": self._frames_seen,
            "width": self._observed_width or self.resolution[0],
            "height": self._observed_height or self.resolution[1],
            "fps": self._observed_fps or float(self.fps),
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------ subscribe
    async def attach(
        self,
        *,
        lossless: bool,
        queue_size: int | None = None,
        name: str = "anon",
    ) -> Subscriber:
        if not self.device_present():
            raise RuntimeError("No RealSense device available")
        loop = asyncio.get_running_loop()
        size = queue_size
        if size is None:
            size = (
                self.DEFAULT_LOSSLESS_QUEUE_SIZE if lossless else self.DEFAULT_LOSSY_QUEUE_SIZE
            )
        sub = Subscriber(name=name, queue=asyncio.Queue(maxsize=size), lossless=lossless, loop=loop)
        with self._sub_lock:
            self._subscribers.append(sub)
            total = len(self._subscribers)
        log.info("LiveCamera: attached subscriber (%s, total=%s)", name, total)
        try:
            await asyncio.to_thread(self._start_thread_if_needed)
        except BaseException:
            with self._sub_lock:
                if sub in self._subscribers:
                    self._subscribers.remove(sub)
            raise
        return sub

    async def detach(self, sub: Subscriber) -> None:
        with self._sub_lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)
            total = len(self._subscribers)
        log.info("LiveCamera: detached subscriber (%s, total=%s)", sub.name, total)
        if total == 0:
            await asyncio.to_thread(self._stop_thread_if_idle)

    async def shutdown(self) -> None:
        with self._sub_lock:
            self._subscribers.clear()
        await asyncio.to_thread(self._stop_thread_if_idle)

    # ------------------------------------------------------------------ thread
    def _make_driver(self) -> ContextManager[object]:
        if self._driver_factory is not None:
            return self._driver_factory()
        from .realsense_live import RealSenseLiveSource

        w, h = self.resolution
        return RealSenseLiveSource(width=w, height=h, fps=self.fps)

    def _start_thread_if_needed(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_event.clear()
        self._driver_error = None
        self._frames_seen = 0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tk_vision-live-camera"
        )
        self._thread.start()
        if not self._started_event.wait(timeout=self.PIPELINE_START_TIMEOUT_S):
            self._stop_event.set()
            self._thread.join(timeout=self.PIPELINE_STOP_TIMEOUT_S)
            self._thread = None
            err = self._driver_error or RuntimeError(
                f"LiveCamera failed to start within {self.PIPELINE_START_TIMEOUT_S}s"
            )
            raise err if isinstance(err, BaseException) else RuntimeError(str(err))
        if self._driver_error:
            self._thread.join(timeout=self.PIPELINE_STOP_TIMEOUT_S)
            self._thread = None
            raise self._driver_error

    def _stop_thread_if_idle(self) -> None:
        with self._sub_lock:
            if self._subscribers:
                return
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.PIPELINE_STOP_TIMEOUT_S)
            self._thread = None

    def _run(self) -> None:
        try:
            with self._make_driver() as src:
                self._observed_width = getattr(src, "width", None) or self.resolution[0]
                self._observed_height = getattr(src, "height", None) or self.resolution[1]
                self._observed_fps = float(getattr(src, "fps", self.fps))
                self._started_event.set()
                log.info(
                    "LiveCamera: pipeline started %sx%s @ %s fps",
                    self._observed_width,
                    self._observed_height,
                    self._observed_fps,
                )
                frames_iter: Iterator[Frame] = src.frames()  # type: ignore[assignment]
                for f in frames_iter:
                    if self._stop_event.is_set():
                        break
                    self._broadcast(f)
                    self._frames_seen += 1
        except BaseException as e:  # noqa: BLE001
            log.exception("LiveCamera driver error: %s", e)
            self._driver_error = e
            self._last_error = str(e)
            self._started_event.set()
        finally:
            self._started_event.set()
            log.info("LiveCamera: pipeline stopped (frames=%s)", self._frames_seen)

    def _broadcast(self, frame: Frame) -> None:
        # The driver may reuse its image buffer between frames; copy once.
        safe = Frame(
            index=frame.index,
            image_bgr=np.ascontiguousarray(frame.image_bgr),
            timestamp_s=frame.timestamp_s,
        )
        with self._sub_lock:
            subs = list(self._subscribers)
        for sub in subs:
            if sub.lossless:
                sub.loop.call_soon_threadsafe(_lossless_put, sub.queue, safe, sub.name)
            else:
                sub.loop.call_soon_threadsafe(_lossy_put, sub.queue, safe)
