from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tk_vision")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the FastAPI server.")
    p_serve.add_argument("--config", default=None, help="Path to YAML config (default: configs/default.yaml).")
    p_serve.add_argument("--host", default=None, help="Bind host (default: from config).")
    p_serve.add_argument("--port", default=None, type=int, help="Port (default: from config, 28000).")
    p_serve.add_argument("--bind", default=None, help='Convenience: --bind 0.0.0.0 with a printed warning.')
    p_serve.add_argument("--no-sam3", action="store_true", help="Skip loading SAM3 (frees VRAM for training).")
    p_serve.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev).")

    p_clean = sub.add_parser(
        "cleanup-stale",
        help="Drop tracks with degenerate (full-frame / edge-touching) masks from clip manifests.",
    )
    p_clean.add_argument("--config", default=None)
    p_clean.add_argument("--clip", default=None, help="Specific clip_id; omit with --all to clean every clip.")
    p_clean.add_argument("--all", action="store_true", help="Clean every clip under data/clips/.")
    p_clean.add_argument("--dry-run", action="store_true", help="Report only; don't mutate manifests or files.")

    p_fetch = sub.add_parser(
        "fetch-weights",
        help="Install / verify a SAM3 checkpoint at sam3_checkpoint_hf/.",
    )
    p_fetch.add_argument(
        "--source",
        choices=("local", "hf", "url"),
        default="local",
        help="local = verify existing dest with the tracker_neck patch canary; "
             "hf = snapshot_download a Hugging Face repo; "
             "url = download a tarball with the sam3_checkpoint_hf/ layout.",
    )
    p_fetch.add_argument("--repo", default=None, help="HF repo id (with --source hf)")
    p_fetch.add_argument("--url", default=None, help="HTTPS URL to a .tar.gz (with --source url)")
    p_fetch.add_argument("--dest", default="./sam3_checkpoint_hf", help="Output directory.")
    p_fetch.add_argument("--force", action="store_true", help="Overwrite a non-empty --dest.")
    p_fetch.add_argument("--sha256", default=None, help="Expected SHA-256 of the tarball (--source url).")
    p_fetch.add_argument("--yes", action="store_true",
                         help="Required for --source hf|url to confirm a multi-GB download.")
    p_fetch.add_argument("--no-verify", action="store_true",
                         help="Skip the post-install canary (not recommended).")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from .app import serve

        serve(
            host=args.host,
            port=args.port,
            config=args.config,
            bind=args.bind,
            no_sam3=args.no_sam3,
            reload=args.reload,
        )
        return 0

    if args.cmd == "cleanup-stale":
        return _cleanup_stale_cli(args)

    if args.cmd == "fetch-weights":
        return _fetch_weights_cli(args)

    parser.print_help()
    return 2


def _fetch_weights_cli(args) -> int:  # noqa: ANN001
    """Install / verify a SAM3 checkpoint.

    Three sources:
      - local: verify the existing dest with the tracker_neck patch canary.
      - hf:    snapshot_download a Hugging Face repo into dest (atomic).
      - url:   download .tar.gz, optional SHA-256 verify, extract into dest.

    Always runs the canary unless --no-verify is set. The canary loads
    Sam3VideoModel.from_pretrained(dest) and asserts that
    Sam3Engine._patch_tracker_neck returns 22; if not, the install is
    rejected (dest reverted to its prior state where possible).
    """
    import os
    import shutil
    from pathlib import Path

    dest = Path(args.dest).resolve()
    src = args.source

    if src in ("hf", "url") and not (args.yes or os.environ.get("TK_VISION_ALLOW_DOWNLOAD") == "1"):
        print(
            "error: --source hf|url will download multi-GB weights. "
            "Pass --yes or set TK_VISION_ALLOW_DOWNLOAD=1 to confirm.",
            file=sys.stderr,
        )
        return 2

    _normalize_proxy_env_for_httpx()

    if src == "local":
        if not (dest / "model.safetensors").is_file():
            print(f"error: {dest}/model.safetensors not found.", file=sys.stderr)
            print("Provide a checkpoint with --source hf or --source url.", file=sys.stderr)
            return 1
    elif src == "hf":
        if not args.repo:
            print("error: --source hf requires --repo <hf_repo_id>.", file=sys.stderr)
            return 2
        if not _check_dest_writable(dest, args.force):
            return 1
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("error: huggingface_hub not installed.", file=sys.stderr)
            return 1
        staging = _staging_dir(dest)
        print(f"snapshot_download({args.repo}) → {staging}")
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(staging),
        )
        _atomic_install(staging, dest)
    elif src == "url":
        if not args.url:
            print("error: --source url requires --url <https://...>", file=sys.stderr)
            return 2
        if not _check_dest_writable(dest, args.force):
            return 1
        return _fetch_weights_url(args.url, dest, args.sha256)

    if args.no_verify:
        print(f"installed (canary skipped) → {dest}")
        return 0
    return _verify_checkpoint(dest)


def _check_dest_writable(dest, force: bool) -> bool:
    """Block overwriting a non-empty dest unless --force."""
    if dest.exists() and next(dest.iterdir(), None) is not None and not force:
        print(f"error: {dest} is not empty. Pass --force to overwrite.", file=sys.stderr)
        return False
    return True


def _normalize_proxy_env_for_httpx() -> None:
    """Normalize proxy env vars so httpx accepts SOCKS proxies.

    Some environments export `socks://...`, but httpx expects
    `socks5://...` (or `socks5h://...`). Rewrite in-process so
    `snapshot_download()` can initialize its HTTP client.
    """
    import os

    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    for key in keys:
        value = os.environ.get(key)
        if not value:
            continue
        if value.startswith("socks://"):
            os.environ[key] = "socks5://" + value[len("socks://") :]
            print(
                f"warning: normalized {key}=socks://... to socks5://... for httpx compatibility",
                file=sys.stderr,
            )


def _staging_dir(dest) -> "Path":
    import shutil
    from pathlib import Path
    staging = Path(dest).with_suffix(".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _atomic_install(staging, dest) -> None:
    """Replace dest with staging atomically."""
    import os
    import shutil
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(staging, dest)


def _fetch_weights_url(url: str, dest, sha256_expected: str | None) -> int:
    import hashlib
    import shutil
    import tarfile
    import tempfile
    from pathlib import Path
    import urllib.request

    dest = Path(dest)
    staging = _staging_dir(dest)

    print(f"downloading {url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=120) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp_path = Path(tmp.name)

    try:
        if sha256_expected:
            h = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            got = h.hexdigest()
            if got != sha256_expected:
                print(f"error: sha256 mismatch (got {got}, expected {sha256_expected})", file=sys.stderr)
                return 1
            print(f"sha256 OK: {got}")
        with tarfile.open(tmp_path, "r:*") as tar:
            tar.extractall(staging, filter="data")
    finally:
        tmp_path.unlink(missing_ok=True)

    cands = list(staging.rglob("model.safetensors"))
    if not cands:
        print("error: no model.safetensors in tarball.", file=sys.stderr)
        return 1
    extracted = cands[0].parent

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(extracted), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    return 0


def _verify_checkpoint(dest) -> int:
    """Load the checkpoint and run the tracker_neck patch canary."""
    import sys as _sys
    from pathlib import Path

    dest = Path(dest)
    try:
        from transformers import Sam3VideoModel
        from .annotate.sam3 import Sam3Engine
        import torch
        model = Sam3VideoModel.from_pretrained(str(dest), dtype=torch.float32)
        n = Sam3Engine._patch_tracker_neck(model)
    except Exception as e:  # noqa: BLE001
        print(f"error: load failed: {e}", file=_sys.stderr)
        return 1
    if n == 22:
        print(f"verified: tracker_neck patch loaded {n}/22 weights from {dest}")
        return 0
    print(
        f"error: tracker_neck patch loaded {n}/22 weights from {dest}.\n"
        f"This checkpoint is incomplete. Try a different --repo or --url, "
        f"or pass --force to overwrite with `tk_vision fetch-weights --source hf --repo <other_repo>`.",
        file=_sys.stderr,
    )
    return 1


def _cleanup_stale_cli(args) -> int:  # noqa: ANN001
    from .config import Settings
    from .data.persistence import ProjectStore
    from .services.label_service import cleanup_clip

    settings = Settings.load(args.config)
    store = ProjectStore(settings.resolve(settings.project.data_root))

    if not args.all and not args.clip:
        print("error: pass --clip <id> or --all", file=sys.stderr)
        return 2

    if args.clip and args.all:
        print("error: pass --clip OR --all, not both", file=sys.stderr)
        return 2

    clip_ids = [args.clip] if args.clip else store.list_clip_ids()
    if not clip_ids:
        print("no clips to scan")
        return 0

    from .services.label_service import scan_degenerate_track_ids

    total_dropped = 0
    for cid in clip_ids:
        clip = store.read_clip(cid)
        before = len(clip.tracks)
        if args.dry_run:
            would_drop = scan_degenerate_track_ids(store, clip)
            print(f"{cid}: would drop {len(would_drop)}/{before} tracks: {would_drop}")
            total_dropped += len(would_drop)
            continue
        _, dropped = cleanup_clip(store, cid)
        print(f"{cid}: dropped {len(dropped)}/{before} tracks: {dropped}")
        total_dropped += len(dropped)

    print(f"total {'would drop' if args.dry_run else 'dropped'}: {total_dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
