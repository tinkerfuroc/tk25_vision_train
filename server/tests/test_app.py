from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tk_vision.app import create_app
from tk_vision.config import Settings


def _client(tmp_path: Path | None = None) -> TestClient:
    s = Settings.load()
    if tmp_path is not None:
        s.project.data_root = str(tmp_path / "data")
        s.repo_root = str(tmp_path)
    return TestClient(create_app(s, load_sam3=False))


def test_health() -> None:
    with _client() as c:
        r = c.get("/api/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["sam3_loaded"] is False


def test_ontology_get() -> None:
    with _client() as c:
        r = c.get("/api/ontology")
        assert r.status_code == 200
        body = r.json()
        assert body["mapping"]
        assert body["prompts"] == list(body["mapping"].keys())
        assert body["labels"] == list(body["mapping"].values())


def test_clips_empty(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/api/clips")
        assert r.status_code == 200
        assert r.json() == []
