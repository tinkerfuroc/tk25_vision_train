from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import cv2

from ..capture.folder import FolderSource
from ..capture.live_camera import LiveCamera, Subscriber
from ..data.manifest import ClipMeta
from ..data.persistence import ProjectStore

log = logging.getLogger("tk_vision.capture")

ProgressCb = Callable[[dict], Awaitable[None]]
FrameCb = Callable[[bytes], Awaitable[None]]


@dataclass
class RecorderState:
    clip_id: str
    requested_seconds: float
    written: int = 0
    started_at: float = field(default_factory=time.time)
    stopped: bool = False
    error: str | None = None


class CaptureService:
    """Orchestrator for capture (live, .bag, folder) and clip persistence.

    Live recording subscribes to the shared `LiveCamera` and writes each
    received frame to disk. The same camera serves the MJPEG preview, so the
    SPA can keep showing the live feed while a clip is being recorded.
    """

    PREVIEW_TARGET_FPS = 12
    PREVIEW_QUALITY = 60

    def __init__(
        self,
        store: ProjectStore,
        *,
        fps: int,
        resolution: tuple[int, int],
        live_camera: LiveCamera,
    ) -> None:
        self.store = store
        self.fps = int(fps)
        self.resolution = resolution
        self.live_camera = live_camera
        self._recorder: RecorderState | None = None
        self._stop_event = asyncio.Event()
        self._progress_subs: dict[str, list[ProgressCb]] = {}
        self._frame_subs: dict[str, list[FrameCb]] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _new_clip_id(prefix: str) -> str:
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_{uuid.uuid4().hex[:6]}"

    def is_recording(self) -> bool:
        return self._recorder is not None and not self._recorder.stopped

    def recorder_state(self) -> RecorderState | None:
        return self._recorder

    async def subscribe_progress(self, clip_id: str, cb: ProgressCb) -> None:
        self._progress_subs.setdefault(clip_id, []).append(cb)

    async def unsubscribe_progress(self, clip_id: str, cb: ProgressCb) -> None:
        if clip_id in self._progress_subs:
            try:
                self._progress_subs[clip_id].remove(cb)
            except ValueError:
                pass

    async def subscribe_frames(self, clip_id: str, cb: FrameCb) -> None:
        self._frame_subs.setdefault(clip_id, []).append(cb)

    async def unsubscribe_frames(self, clip_id: str, cb: FrameCb) -> None:
        if clip_id in self._frame_subs:
            try:
                self._frame_subs[clip_id].remove(cb)
            except ValueError:
                pass

    async def _emit_progress(self, clip_id: str, payload: dict) -> None:
        for cb in list(self._progress_subs.get(clip_id, ())):
            try:
                await cb(payload)
            except Exception as e:  # noqa: BLE001
                log.warning("progress subscriber failed: %s", e)

    async def _emit_frame(self, clip_id: str, jpeg: bytes) -> None:
        for cb in list(self._frame_subs.get(clip_id, ())):
            try:
                await cb(jpeg)
            except Exception as e:  # noqa: BLE001
                log.warning("frame subscriber failed: %s", e)

    # ------------------------------------------------------------------ import
    def import_folder(self, src: str | Path) -> ClipMeta:
        src_path = Path(src).expanduser().resolve()
        with FolderSource(src_path, fps=self.fps) as fs:
            clip_id = self._new_clip_id("folder")
            cdir = self.store.clip_dir(clip_id)
            (cdir / "frames").mkdir(parents=True, exist_ok=True)
            count = 0
            for f in fs.frames():
                cv2.imwrite(str(cdir / "frames" / f"{count:06d}.jpg"), f.image_bgr)
                count += 1
            meta = ClipMeta(
                clip_id=clip_id,
                source="folder",
                folder_path=str(src_path),
                fps=fs.fps,
                width=fs.width,
                height=fs.height,
                frame_count=count,
                created_at=time.time(),
            )
            self.store.write_meta(meta)
            return meta

    def import_bag(self, bag_path: str | Path) -> ClipMeta:
        from ..capture.realsense_bag import RealSenseBagSource

        bp = Path(bag_path).expanduser().resolve()
        if not bp.exists():
            raise FileNotFoundError(bp)
        clip_id = self._new_clip_id("bag")
        cdir = self.store.clip_dir(clip_id)
        (cdir / "frames").mkdir(parents=True, exist_ok=True)
        copied_bag = cdir / "raw.bag"
        shutil.copy2(bp, copied_bag)
        with RealSenseBagSource(copied_bag) as bs:
            count = 0
            for f in bs.frames():
                cv2.imwrite(str(cdir / "frames" / f"{count:06d}.jpg"), f.image_bgr)
                count += 1
            meta = ClipMeta(
                clip_id=clip_id,
                source="bag",
                bag_path=str(copied_bag),
                fps=bs.fps,
                width=bs.width,
                height=bs.height,
                frame_count=count,
                created_at=time.time(),
            )
        self.store.write_meta(meta)
        return meta

    # ------------------------------------------------------------------ record
    async def start_record(self, *, max_seconds: float) -> str:
        if self.is_recording():
            raise RuntimeError("Another recording is already in progress.")
        if not self.live_camera.device_present():
            raise RuntimeError("No RealSense device available.")
        clip_id = self._new_clip_id("live")
        cdir = self.store.clip_dir(clip_id)
        (cdir / "frames").mkdir(parents=True, exist_ok=True)
        self._recorder = RecorderState(clip_id=clip_id, requested_seconds=float(max_seconds))
        self._stop_event = asyncio.Event()
        asyncio.create_task(self._record_loop(clip_id, max_seconds, cdir))
        return clip_id

    async def stop_record(self) -> None:
        if self._recorder is None:
            return
        self._stop_event.set()

    async def _record_loop(self, clip_id: str, max_seconds: float, cdir: Path) -> None:
        rec = self._recorder
        assert rec is not None
        loop = asyncio.get_running_loop()
        preview_every = max(1, int(self.fps / self.PREVIEW_TARGET_FPS))
        deadline = time.time() + max_seconds
        sub: Subscriber | None = None
        width = self.resolution[0]
        height = self.resolution[1]
        observed_fps = float(self.fps)

        try:
            try:
                sub = await self.live_camera.attach(
                    lossless=True, name=f"rec:{clip_id}"
                )
            except Exception as e:  # noqa: BLE001
                log.exception("recording attach failed: %s", e)
                rec.error = str(e)
                rec.stopped = True
                await self._emit_progress(clip_id, {"event": "error", "detail": str(e)})
                return

            await self._emit_progress(clip_id, {"event": "started", "clip_id": clip_id})
            count = 0

            while True:
                if self._stop_event.is_set() or time.time() >= deadline:
                    break
                try:
                    frame = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if count == 0:
                    height, width = frame.image_bgr.shape[:2]
                    cam_stats = self.live_camera.stats()
                    observed_fps = float(cam_stats.get("fps", self.fps))

                jpg_path = str(cdir / "frames" / f"{count:06d}.jpg")
                await loop.run_in_executor(None, cv2.imwrite, jpg_path, frame.image_bgr)
                count += 1
                rec.written = count

                if count <= 5 or count % 5 == 0:
                    await self._emit_progress(
                        clip_id,
                        {
                            "event": "frame",
                            "written": count,
                            "elapsed": time.time() - rec.started_at,
                        },
                    )

                if count == 1 or count % preview_every == 0:
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame.image_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.PREVIEW_QUALITY],
                    )
                    if ok:
                        await self._emit_frame(clip_id, buf.tobytes())

            meta = ClipMeta(
                clip_id=clip_id,
                source="live",
                bag_path=None,
                fps=observed_fps,
                width=width,
                height=height,
                frame_count=count,
                created_at=rec.started_at,
            )
            self.store.write_meta(meta)
            rec.stopped = True
            await self._emit_progress(
                clip_id, {"event": "completed", "frame_count": count}
            )
        except Exception as e:  # noqa: BLE001
            log.exception("recording loop failed: %s", e)
            if rec is not None:
                rec.error = str(e)
                rec.stopped = True
            await self._emit_progress(clip_id, {"event": "error", "detail": str(e)})
        finally:
            if sub is not None:
                await self.live_camera.detach(sub)
            await asyncio.sleep(0)
            self._recorder = None
