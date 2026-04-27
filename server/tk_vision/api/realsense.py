from __future__ import annotations

import asyncio
from typing import AsyncIterator

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/realsense", tags=["realsense"])


class RealSenseStatus(BaseModel):
    available: bool
    busy: bool
    recording_clip_id: str | None
    running: bool
    subscribers: int
    width: int
    height: int
    fps: float


@router.get("/status", response_model=RealSenseStatus)
async def status(request: Request) -> RealSenseStatus:
    cap = getattr(request.app.state, "capture", None)
    cam = getattr(request.app.state, "live_camera", None)
    rec = cap.recorder_state() if cap else None
    cam_stats = cam.stats() if cam else {}
    return RealSenseStatus(
        available=bool(cam and cam.device_present()),
        busy=cap.is_recording() if cap else False,
        recording_clip_id=rec.clip_id if rec else None,
        running=bool(cam_stats.get("running", False)),
        subscribers=int(cam_stats.get("subscribers", 0)),
        width=int(cam_stats.get("width", 0)),
        height=int(cam_stats.get("height", 0)),
        fps=float(cam_stats.get("fps", 0.0)),
    )


def _mjpeg_part(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
        jpeg_bytes + b"\r\n"
    )


@router.get("/stream.mjpg")
async def stream_mjpeg(request: Request) -> StreamingResponse:
    cam = getattr(request.app.state, "live_camera", None)
    if cam is None:
        raise HTTPException(503, "LiveCamera not initialized")
    if not cam.device_present():
        raise HTTPException(503, "No RealSense device available")

    try:
        sub = await cam.attach(lossless=False, name="preview")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"RealSense open failed: {e}") from e

    loop = asyncio.get_running_loop()

    async def gen() -> AsyncIterator[bytes]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    frame = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                ok, buf = await loop.run_in_executor(
                    None,
                    lambda img=frame.image_bgr: cv2.imencode(
                        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    ),
                )
                if not ok:
                    continue
                yield _mjpeg_part(buf.tobytes())
        finally:
            await cam.detach(sub)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
