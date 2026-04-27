"""Regression tests for path-traversal hardening.

Covers `safe_id`, ProjectStore path containment, and `export_run` traversal
rejection. Codex adversarial review (2026-04-26) flagged a critical
`run_id="."` / `".."` flaw that let `shutil.rmtree(run_dir)` wipe data_root;
these tests lock the patch in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tk_vision.data.persistence import ProjectStore, _assert_contained, safe_id


# ---------------------------------------------------------------- safe_id

@pytest.mark.parametrize(
    "bad",
    [
        "",
        ".",
        "..",
        ".hidden",
        "foo/bar",
        "foo\\bar",
        "\x00abc",
        "a" * 256,
        "../etc",
        "/abs",
        "with space",
    ],
)
def test_safe_id_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        safe_id(bad)


@pytest.mark.parametrize(
    "good",
    [
        "clip_001",
        "20260426_193015_folder_a1b2c3",
        "smoke-run.v2",
        "A",
        "a1",
        "weights.bf16",
    ],
)
def test_safe_id_accepts_good(good: str) -> None:
    assert safe_id(good) == good


# ---------------------------------------------------------------- containment

def test_clip_dir_rejects_traversal(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.clip_dir("..")
    with pytest.raises(ValueError):
        store.clip_dir(".")
    with pytest.raises(ValueError):
        store.clip_dir("../outside")
    with pytest.raises(ValueError):
        store.clip_dir("a/b")


def test_clip_dir_accepts_normal_id(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    p = store.clip_dir("clip_001")
    assert p.parent.resolve() == store.clips_dir.resolve()


def test_run_dir_rejects_traversal(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.run_dir("..")
    with pytest.raises(ValueError):
        store.run_dir(".")


def test_assert_contained_rejects_escape(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    with pytest.raises(ValueError):
        _assert_contained(sibling, parent)
    with pytest.raises(ValueError):
        _assert_contained(parent, parent)  # equal-to-parent is also rejected


# ---------------------------------------------------------------- export_run

def test_export_run_rejects_dot_traversal(tmp_path: Path) -> None:
    """The original critical bug: run_id='.' / '..' with overwrite=True
    would rmtree(runs_dir) or rmtree(data_root). Sentinel files must survive."""
    from tk_vision.services.export_service import export_run

    store = ProjectStore(tmp_path)
    sentinel = store.runs_dir / "untouchable_run"
    sentinel.mkdir()
    (sentinel / "marker").write_text("do not delete")

    for bad in ("..", ".", "..//evil", "/abs"):
        with pytest.raises(ValueError):
            export_run(store, run_id=bad, overwrite=True)
        assert sentinel.is_dir()
        assert (sentinel / "marker").read_text() == "do not delete"


def test_data_root_intact_after_traversal_attempt(tmp_path: Path) -> None:
    """Confirm `data_root` itself survives a `..` attack."""
    from tk_vision.services.export_service import export_run

    store = ProjectStore(tmp_path)
    canary = tmp_path / "canary.txt"
    canary.write_text("alive")

    with pytest.raises(ValueError):
        export_run(store, run_id="..", overwrite=True)

    assert canary.exists()
    assert canary.read_text() == "alive"
    assert store.runs_dir.is_dir()
    assert store.clips_dir.is_dir()


# ---------------------------------------------------------------- HTTP layer

def test_http_delete_clip_traversal_returns_400(tmp_path: Path) -> None:
    """`DELETE /api/clips/..` must NOT delete data_root. ValueError handler
    in app.py converts the safe_id failure to HTTP 400."""
    from fastapi.testclient import TestClient

    from tk_vision.app import create_app
    from tk_vision.config import Settings

    s = Settings.load()
    s.project.data_root = str(tmp_path / "data")
    s.repo_root = str(tmp_path)
    canary = tmp_path / "data" / "canary.txt"

    with TestClient(create_app(s, load_sam3=False)) as c:
        canary.parent.mkdir(parents=True, exist_ok=True)
        canary.write_text("alive")
        # Encoded so Starlette/httpx don't normalize before sending.
        r = c.delete("/api/clips/%2E%2E")
        assert r.status_code == 400
        assert canary.exists()
        assert canary.read_text() == "alive"
