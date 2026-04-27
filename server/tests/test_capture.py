from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tk_vision.app import create_app
from tk_vision.config import Settings


def _settings(tmp_path: Path) -> Settings:
    s = Settings.load()
    s.project.data_root = str(tmp_path / "data")
    s.repo_root = str(tmp_path)
    return s


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_settings(tmp_path), load_sam3=False))


def _make_folder_clip(folder: Path, n: int = 6, w: int = 64, h: int = 48) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((h, w, 3), i * 30 % 255, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:03d}.jpg"), img)
    return folder


def test_folder_import_lists_and_serves(tmp_path: Path) -> None:
    src = _make_folder_clip(tmp_path / "src", n=5, w=80, h=60)
    with _client(tmp_path) as c:
        r = c.post("/api/clips/import", json={"folder_path": str(src)})
        assert r.status_code == 200, r.text
        clip = r.json()
        assert clip["frame_count"] == 5
        assert clip["width"] == 80 and clip["height"] == 60

        listing = c.get("/api/clips").json()
        assert any(x["clip_id"] == clip["clip_id"] for x in listing)

        r = c.get(f"/api/clips/{clip['clip_id']}")
        assert r.status_code == 200

        r = c.get(f"/api/clips/{clip['clip_id']}/frames/0")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/jpeg")

        r = c.get(f"/api/clips/{clip['clip_id']}/frames/99")
        assert r.status_code == 404

        r = c.delete(f"/api/clips/{clip['clip_id']}")
        assert r.status_code == 200
        assert c.get(f"/api/clips/{clip['clip_id']}").status_code == 404


def test_meta_persisted(tmp_path: Path) -> None:
    src = _make_folder_clip(tmp_path / "src", n=3)
    s = _settings(tmp_path)
    with TestClient(create_app(s, load_sam3=False)) as c:
        clip = c.post("/api/clips/import", json={"folder_path": str(src)}).json()
    meta_path = Path(s.resolve(s.project.data_root)) / "clips" / clip["clip_id"] / "meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert meta["source"] == "folder"
    assert meta["frame_count"] == 3


def test_realsense_status_no_hardware(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/api/realsense/status")
        assert r.status_code == 200
        body = r.json()
        assert body["busy"] is False
        # `available` depends on whether pyrealsense2 is installed and a device is plugged in.
        assert isinstance(body["available"], bool)


def test_record_without_realsense_fails_cleanly(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.post("/api/clips/record", json={"max_seconds": 0.5})
        # Either pyrealsense2 isn't installed (record returns 200 + a clip_id but the bg task
        # records nothing) or it is installed without a device (also fine). We only assert
        # the request itself doesn't 500.
        assert r.status_code in (200, 503), r.text
