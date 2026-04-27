"""YOLO-seg replay; persists <runs>/<run_id>/inference/<clip_id>.json.

Detection schema (one entry per detection):
    {
        "class_id": int,
        "label": str,
        "score": float,
        "polygon_norm": [x0, y0, x1, y1, ...],   # YOLO-normalized
        "bbox_norm": [x0, y0, x1, y1],            # xyxy normalized
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from ..data.persistence import ProjectStore
from .polygons import normalize_polygon
from .train_service import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    _PROGRESS_QUEUE_MAX,
    _put_with_drop,
)

log = logging.getLogger("tk_vision.infer")

InferStatus = Literal["pending", "running", "done", "error", "cancelled"]


@dataclass
class InferJob:
    job_id: str
    run_id: str
    clip_id: str
    weights_path: str
    conf: float
    iou: float
    status: InferStatus = "pending"
    error: Optional[str] = None
    done_frames: int = 0
    total_frames: int = 0
    predictions_path: Optional[str] = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=512))
    progress: asyncio.Queue[dict] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_PROGRESS_QUEUE_MAX)
    )
    cancel: asyncio.Event = field(default_factory=asyncio.Event)


class InferManager:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self.jobs: dict[str, InferJob] = {}

    def list_jobs(self, run_id: str | None = None) -> list[InferJob]:
        if run_id is None:
            return list(self.jobs.values())
        return [j for j in self.jobs.values() if j.run_id == run_id]

    def get(self, job_id: str) -> InferJob | None:
        return self.jobs.get(job_id)

    def start(
        self,
        *,
        run_id: str,
        clip_id: str,
        weights_path: Path,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> InferJob:
        if not weights_path.exists():
            raise FileNotFoundError(f"weights not found: {weights_path}")
        clip_dir = self.store.clip_dir(clip_id)
        if not clip_dir.exists():
            raise FileNotFoundError(f"clip not found: {clip_id}")
        run_dir = self.store.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run not found: {run_id}")

        job = InferJob(
            job_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            clip_id=clip_id,
            weights_path=str(weights_path),
            conf=conf,
            iou=iou,
        )
        self.jobs[job.job_id] = job
        asyncio.create_task(self._run(job))
        return job

    def cancel(self, job_id: str) -> InferJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in ACTIVE_STATUSES:
            job.cancel.set()
        return job

    async def _run(self, job: InferJob) -> None:
        try:
            job.status = "running"
            await _put_with_drop(job.progress, {"event": "started"})
            loop = asyncio.get_running_loop()
            await asyncio.to_thread(self._run_blocking, job, loop)
            if job.cancel.is_set():
                job.status = "cancelled"
                await _put_with_drop(job.progress, {"event": "cancelled"})
            else:
                job.status = "done"
                await _put_with_drop(
                    job.progress,
                    {"event": "done", "predictions_path": job.predictions_path},
                )
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = str(e)
            await _put_with_drop(job.progress, {"event": "error", "detail": str(e)})

    def _run_blocking(self, job: InferJob, loop: asyncio.AbstractEventLoop) -> None:
        from ultralytics import YOLO

        clip = self.store.read_clip(job.clip_id)
        deleted = set(clip.deleted_frames)
        frames_dir = self.store.frames_dir(job.clip_id)
        all_frames = sorted(frames_dir.glob("*.jpg"))
        targets = [p for p in all_frames if int(p.stem) not in deleted]
        job.total_frames = len(targets)

        model = YOLO(job.weights_path)
        names: dict[int, str] = getattr(model, "names", {}) or {}
        per_frame: dict[int, list[dict]] = {}

        for path in targets:
            if job.cancel.is_set():
                break
            fi = int(path.stem)
            results = model.predict(
                source=str(path),
                conf=job.conf,
                iou=job.iou,
                verbose=False,
            )
            detections: list[dict] = []
            r = results[0] if results else None
            if r is not None and r.masks is not None and r.boxes is not None:
                w = clip.width or r.orig_shape[1]
                h = clip.height or r.orig_shape[0]
                cls = r.boxes.cls.tolist() if r.boxes.cls is not None else []
                confs = r.boxes.conf.tolist() if r.boxes.conf is not None else []
                xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
                xy_polys = r.masks.xy
                for c, sc, bb, poly in zip(cls, confs, xyxy, xy_polys):
                    poly_flat = [float(v) for pt in poly for v in pt]
                    poly_norm = normalize_polygon(poly_flat, w, h)
                    bbox_norm = [bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h]
                    cid = int(c)
                    detections.append(
                        {
                            "class_id": cid,
                            "label": names.get(cid, f"class_{cid}"),
                            "score": float(sc),
                            "polygon_norm": poly_norm,
                            "bbox_norm": bbox_norm,
                        }
                    )
            per_frame[fi] = detections
            job.done_frames += 1
            asyncio.run_coroutine_threadsafe(
                _put_with_drop(
                    job.progress,
                    {
                        "event": "frame",
                        "frame_idx": fi,
                        "done": job.done_frames,
                        "total": job.total_frames,
                        "detections": len(detections),
                    },
                ),
                loop,
            )

        out_dir = self.store.runs_dir / job.run_id / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job.clip_id}.json"
        out_path.write_text(
            json.dumps(
                {
                    "run_id": job.run_id,
                    "clip_id": job.clip_id,
                    "weights_path": job.weights_path,
                    "conf": job.conf,
                    "iou": job.iou,
                    "frame_count": clip.frame_count,
                    "deleted_frames": list(clip.deleted_frames),
                    "predictions": {str(k): v for k, v in sorted(per_frame.items())},
                },
                indent=2,
            )
        )
        job.predictions_path = str(out_path)


def load_predictions(store: ProjectStore, run_id: str, clip_id: str) -> dict | None:
    p = store.runs_dir / run_id / "inference" / f"{clip_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


__all__ = [
    "InferJob",
    "InferManager",
    "InferStatus",
    "TERMINAL_STATUSES",
    "load_predictions",
]
