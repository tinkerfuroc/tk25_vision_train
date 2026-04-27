"""Tests for train service: subprocess lifecycle without invoking real YOLO."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from tk_vision.config import Settings
from tk_vision.services.train_service import TrainJob, TrainManager


def _seed_run_dir(tmp: Path, run_id: str) -> Path:
    rd = tmp / "data" / "runs" / run_id
    (rd / "images" / "train").mkdir(parents=True, exist_ok=True)
    (rd / "data.yaml").write_text(
        "path: x\ntrain: images/train\nval: images/val\nnc: 1\nnames: {0: a}\n"
    )
    return rd


def _settings(tmp: Path) -> Settings:
    s = Settings()
    s.repo_root = str(tmp)
    s.train.project_dir = "./runs"
    return s


def _stub_runner(tmp: Path, exit_code: int = 0, sleep_s: float = 0.0) -> Path:
    stub = tmp / "stub_runner.py"
    stub.write_text(
        f"""
import sys, time
print('[stub] hello', flush=True)
print('[stub] more', flush=True)
time.sleep({sleep_s})
print('[stub] bye', flush=True)
sys.exit({exit_code})
"""
    )
    return stub


def _make_job(tmp: Path, mgr: TrainManager, *, run_id: str, stub: Path) -> TrainJob:
    job = TrainJob(
        job_id=run_id + "_x",
        run_id=run_id,
        cmd=[sys.executable, str(stub)],
        project_dir=tmp / "runs",
        name=run_id + "_x",
    )
    mgr.jobs[job.job_id] = job
    return job


async def _wait_terminal(job: TrainJob, *, timeout_s: float = 10.0) -> None:
    terminal = {"done", "error", "cancelled"}
    deadline = asyncio.get_event_loop().time() + timeout_s
    while job.status not in terminal:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"job stuck in status={job.status!r}")
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_train_job_lifecycle_with_stub(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    mgr = TrainManager(s)
    stub = _stub_runner(tmp_path, exit_code=0, sleep_s=0.1)
    job = _make_job(tmp_path, mgr, run_id="r2", stub=stub)
    task = asyncio.create_task(mgr._run(job))
    await _wait_terminal(job)
    await task
    assert job.status == "done", job.error
    assert job.return_code == 0
    log_lines = list(job.log_lines)
    assert any("hello" in line for line in log_lines)
    assert any("bye" in line for line in log_lines)


@pytest.mark.asyncio
async def test_train_job_cancel(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    mgr = TrainManager(s)
    stub = _stub_runner(tmp_path, exit_code=0, sleep_s=10.0)
    job = _make_job(tmp_path, mgr, run_id="r3", stub=stub)
    task = asyncio.create_task(mgr._run(job))
    await asyncio.sleep(0.3)
    mgr.cancel(job.job_id)
    await _wait_terminal(job)
    await task
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_train_job_failure_propagates_return_code(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    mgr = TrainManager(s)
    stub = _stub_runner(tmp_path, exit_code=2)
    job = _make_job(tmp_path, mgr, run_id="r4", stub=stub)
    task = asyncio.create_task(mgr._run(job))
    await _wait_terminal(job)
    await task
    assert job.status == "error"
    assert job.return_code == 2


def test_train_start_rejects_missing_data_yaml(tmp_path: Path) -> None:
    rd = tmp_path / "data" / "runs" / "rmissing"
    rd.mkdir(parents=True)
    s = _settings(tmp_path)
    mgr = TrainManager(s)
    with pytest.raises(FileNotFoundError):
        mgr.start(run_id="rmissing", run_dir=rd, cfg=s.train)


@pytest.mark.asyncio
async def test_train_start_succeeds_with_data_yaml(tmp_path: Path) -> None:
    rd = _seed_run_dir(tmp_path, "rok")
    s = _settings(tmp_path)
    mgr = TrainManager(s)
    job = mgr.start(run_id="rok", run_dir=rd, cfg=s.train)
    assert job.run_id == "rok"
    assert job.cmd[0] == sys.executable
    mgr.cancel(job.job_id)
    await _wait_terminal(job)


def test_progress_queue_drops_oldest_when_full(tmp_path: Path) -> None:
    """Drop-oldest when the WS consumer is slow; assert the bound is honored."""
    from tk_vision.services.train_service import _PROGRESS_QUEUE_MAX, _put_with_drop

    async def go() -> None:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=_PROGRESS_QUEUE_MAX)
        for i in range(_PROGRESS_QUEUE_MAX + 50):
            await _put_with_drop(q, {"event": "log", "i": i})
        assert q.qsize() == _PROGRESS_QUEUE_MAX
        first = q.get_nowait()
        # Oldest 50 dropped → first surviving event has i == 50.
        assert first["i"] == 50

    asyncio.run(go())
