"""Background YOLO-seg training jobs.

Spawns the training run in a subprocess so a long fit doesn't block the
event loop and so cancellation is just a SIGTERM. Each job carries an
asyncio.Queue of stdout lines; the WS endpoint drains it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from ..config import Settings, TrainCfg

log = logging.getLogger("tk_vision.train")

TrainStatus = Literal["pending", "running", "done", "error", "cancelled"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "error", "cancelled"})
ACTIVE_STATUSES: frozenset[str] = frozenset({"pending", "running"})
_TERMINAL = TERMINAL_STATUSES  # legacy alias
_LOG_RING_MAX = 2000
_PROGRESS_QUEUE_MAX = 512


def _bounded_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=_PROGRESS_QUEUE_MAX)


@dataclass
class TrainJob:
    job_id: str
    run_id: str
    cmd: list[str]
    project_dir: Path
    name: str
    status: TrainStatus = "pending"
    error: Optional[str] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    return_code: Optional[int] = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_RING_MAX))
    progress: asyncio.Queue[dict] = field(default_factory=_bounded_queue)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    process: Optional[asyncio.subprocess.Process] = None
    metrics_cache: Optional[dict] = None


async def _put_with_drop(queue: asyncio.Queue, evt: dict) -> None:
    """Drop-oldest if the WS consumer falls behind."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(evt)


class TrainManager:
    """Per-app singleton tracking active + recent training jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, TrainJob] = {}

    def list_jobs(self, run_id: str | None = None) -> list[TrainJob]:
        if run_id is None:
            return list(self.jobs.values())
        return [j for j in self.jobs.values() if j.run_id == run_id]

    def get(self, job_id: str) -> TrainJob | None:
        return self.jobs.get(job_id)

    def start(
        self,
        *,
        run_id: str,
        run_dir: Path,
        cfg: TrainCfg,
        device: str = "",
    ) -> TrainJob:
        data_yaml = run_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml missing at {data_yaml} — export first")
        project_dir = self.settings.resolve(cfg.project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        name = f"{run_id}_{uuid.uuid4().hex[:6]}"
        cmd = [
            sys.executable,
            "-m",
            "tk_vision._train_runner",
            "--data",
            str(data_yaml),
            "--project",
            str(project_dir),
            "--name",
            name,
            "--base-weights",
            cfg.base_weights,
            "--epochs",
            str(cfg.epochs),
            "--imgsz",
            str(cfg.imgsz),
            "--batch",
            str(cfg.batch),
            "--patience",
            str(cfg.patience),
        ]
        if device:
            cmd += ["--device", device]
        job = TrainJob(
            job_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            cmd=cmd,
            project_dir=project_dir,
            name=name,
        )
        self.jobs[job.job_id] = job
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: TrainJob) -> None:
        try:
            job.status = "running"
            await _put_with_drop(job.progress, {"event": "started", "cmd": shlex.join(job.cmd)})
            proc = await asyncio.create_subprocess_exec(
                *job.cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            job.process = proc
            job.pid = proc.pid
            try:
                job.pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                job.pgid = None

            cancelled = await self._pump_lines(proc, job)
            if cancelled:
                self._terminate(job)
                await proc.wait()
                job.return_code = proc.returncode
                job.status = "cancelled"
                await _put_with_drop(job.progress, {"event": "cancelled"})
                return

            rc = await proc.wait()
            job.return_code = rc
            if rc == 0:
                job.status = "done"
                await _put_with_drop(job.progress, {"event": "done", "return_code": 0})
            else:
                job.status = "error"
                job.error = f"return_code={rc}"
                await _put_with_drop(job.progress, {"event": "error", "return_code": rc})
        except Exception as e:  # noqa: BLE001
            self._terminate(job)
            if job.process is not None:
                try:
                    await asyncio.wait_for(job.process.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            job.status = "error"
            job.error = str(e)
            await _put_with_drop(job.progress, {"event": "error", "detail": str(e)})

    async def _pump_lines(self, proc: asyncio.subprocess.Process, job: TrainJob) -> bool:
        """Read subprocess stdout; return True if cancel fired mid-stream."""
        cancel_task = asyncio.create_task(job.cancel.wait())
        try:
            while True:
                line_task = asyncio.create_task(proc.stdout.readline())
                done, _pending = await asyncio.wait(
                    {line_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_task in done:
                    if not line_task.done():
                        line_task.cancel()
                    return True
                line = line_task.result()
                if not line:
                    return False
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                job.log_lines.append(text)
                await _put_with_drop(job.progress, {"event": "log", "line": text})
        finally:
            cancel_task.cancel()

    def _terminate(self, job: TrainJob) -> None:
        if job.pgid is None:
            return
        try:
            os.killpg(job.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def cancel(self, job_id: str) -> TrainJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in ACTIVE_STATUSES:
            return job
        job.cancel.set()
        return job
