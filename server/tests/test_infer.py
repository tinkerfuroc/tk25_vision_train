"""Tests for inference service: lifecycle without invoking real YOLO."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tk_vision.data.manifest import Clip, ClipMeta
from tk_vision.data.persistence import ProjectStore
from tk_vision.services.infer_service import InferManager, load_predictions


class _FakeBoxes:
    def __init__(self, cls, conf, xyxy) -> None:
        self.cls = np.array(cls)
        self.conf = np.array(conf)
        self.xyxy = np.array(xyxy)


class _FakeMasks:
    def __init__(self, polys) -> None:
        self.xy = polys


class _FakeResult:
    def __init__(self, *, names, cls, conf, xyxy, polys, w: int, h: int) -> None:
        self.names = names
        self.boxes = _FakeBoxes(cls, conf, xyxy)
        self.masks = _FakeMasks(polys)
        self.orig_shape = (h, w)


class _FakeYOLO:
    """Stands in for `ultralytics.YOLO`. Returns one detection per frame."""

    def __init__(self, weights_path: str) -> None:
        self.names = {0: "thing"}

    def predict(self, source: str, conf: float, iou: float, verbose: bool = False):
        return [
            _FakeResult(
                names=self.names,
                cls=[0],
                conf=[0.9],
                xyxy=[[10.0, 10.0, 50.0, 40.0]],
                polys=[np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]])],
                w=64,
                h=48,
            )
        ]


@pytest.fixture(autouse=True)
def _stub_ultralytics(monkeypatch: pytest.MonkeyPatch):
    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    yield


def _seed(tmp: Path) -> tuple[ProjectStore, str]:
    store = ProjectStore(tmp)
    cid = "clipA"
    cdir = store.clip_dir(cid)
    fdir = cdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(3):
        cv2.imwrite(str(fdir / f"{i:06d}.jpg"), img)
    meta = ClipMeta(
        clip_id=cid, source="folder", bag_path=None, folder_path=None,
        fps=30.0, width=64, height=48, frame_count=3, created_at=0.0, intrinsics=None,
    )
    store.write_meta(meta)
    store.write_clip(Clip(**meta.model_dump()))

    run_id = "r1"
    (store.runs_dir / run_id).mkdir(parents=True, exist_ok=True)
    return store, cid


def _weights(tmp: Path) -> Path:
    p = tmp / "weights.pt"
    p.write_bytes(b"\x00")
    return p


@pytest.mark.asyncio
async def test_infer_lifecycle_persists_predictions(tmp_path: Path) -> None:
    store, cid = _seed(tmp_path)
    weights = _weights(tmp_path)
    mgr = InferManager(store)
    job = mgr.start(run_id="r1", clip_id=cid, weights_path=weights)
    deadline = asyncio.get_event_loop().time() + 10.0
    while job.status not in ("done", "error", "cancelled"):
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(job.status)
        await asyncio.sleep(0.05)
    assert job.status == "done", job.error
    assert job.done_frames == 3
    data = load_predictions(store, "r1", cid)
    assert data is not None
    assert data["frame_count"] == 3
    assert "predictions" in data
    assert all(len(p) == 1 for p in data["predictions"].values())
    one = list(data["predictions"].values())[0][0]
    assert one["label"] == "thing"
    assert 0.0 <= one["bbox_norm"][0] <= 1.0


@pytest.mark.asyncio
async def test_infer_skips_deleted_frames(tmp_path: Path) -> None:
    store, cid = _seed(tmp_path)
    clip = store.read_clip(cid)
    clip.deleted_frames = [1]
    store.write_clip(clip)
    weights = _weights(tmp_path)
    mgr = InferManager(store)
    job = mgr.start(run_id="r1", clip_id=cid, weights_path=weights)
    deadline = asyncio.get_event_loop().time() + 10.0
    while job.status not in ("done", "error", "cancelled"):
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(job.status)
        await asyncio.sleep(0.05)
    assert job.done_frames == 2
    data = load_predictions(store, "r1", cid)
    assert set(data["predictions"].keys()) == {"0", "2"}


def test_infer_rejects_missing_weights(tmp_path: Path) -> None:
    store, cid = _seed(tmp_path)
    mgr = InferManager(store)
    with pytest.raises(FileNotFoundError):
        mgr.start(run_id="r1", clip_id=cid, weights_path=tmp_path / "nope.pt")


def test_infer_rejects_unknown_clip(tmp_path: Path) -> None:
    store, _ = _seed(tmp_path)
    weights = _weights(tmp_path)
    mgr = InferManager(store)
    with pytest.raises(FileNotFoundError):
        mgr.start(run_id="r1", clip_id="ghost", weights_path=weights)
