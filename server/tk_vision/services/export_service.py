"""YOLO-seg dataset export.

Materializes one or more clips' frames + per-track masks into a YOLO-seg
training dataset on disk:

    <runs_dir>/<run_id>/
        data.yaml
        images/{train,val}/<clip_id>__<frame_idx>.jpg
        labels/{train,val}/<clip_id>__<frame_idx>.txt

Two split modes:
    per_clip_split=True  → each clip is whole-assigned to train OR val (default).
                           Avoids per-frame leakage on temporally adjacent frames.
    per_clip_split=False → frames pooled across clips and shuffled.

Multi-contour polygon writing: each disconnected component of the mask
becomes its own YOLO-seg polygon line, so donut/holes/two-piece masks
are not silently collapsed to "largest contour only".
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import Settings, load_ontology, ontology_hash
from ..data.manifest import Clip, Track
from ..data.persistence import ProjectStore, _assert_contained, mask_path_for, read_mask, safe_id
from .polygons import mask_to_polygons, normalize_polygon


@dataclass
class SkippedFrame:
    clip_id: str
    frame_idx: int
    reason: str


@dataclass
class ExportStats:
    run_id: str
    train_frames: int = 0
    val_frames: int = 0
    train_polygons: int = 0
    val_polygons: int = 0
    skipped_frames: list[SkippedFrame] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return self.train_frames + self.val_frames


# Polygon helpers are shared with augment_service via .polygons.
_mask_to_polygons = mask_to_polygons
_normalize_polygon = normalize_polygon


def _frame_polygons_for_track(
    track: Track, frame_idx: int, clip_dir: Path
) -> list[list[float]]:
    if frame_idx not in track.masks:
        return []
    p = mask_path_for(clip_dir, track.track_id, frame_idx)
    if not p.exists():
        return []
    mask = read_mask(p)
    return _mask_to_polygons(mask)


def _split_per_clip(
    clip_ids: list[str], train_ratio: float, rng: random.Random
) -> tuple[set[str], set[str]]:
    ids = list(clip_ids)
    rng.shuffle(ids)
    split = max(1, int(round(len(ids) * train_ratio))) if len(ids) > 1 else len(ids)
    return set(ids[:split]), set(ids[split:])


def _split_per_frame(
    items: list[tuple[str, int]], train_ratio: float, rng: random.Random
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    pool = list(items)
    rng.shuffle(pool)
    split = max(1, int(round(len(pool) * train_ratio))) if len(pool) > 1 else len(pool)
    return set(pool[:split]), set(pool[split:])


def _write_data_yaml(
    run_dir: Path,
    classes: list[str],
    *,
    ontology_path: str | None,
    ontology_hash_str: str | None,
) -> None:
    lines: list[str] = []
    lines.append(f"path: {run_dir.resolve()}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append(f"nc: {len(classes)}")
    name_map = ", ".join(f"{i}: {c}" for i, c in enumerate(classes))
    lines.append(f"names: {{{name_map}}}")
    if ontology_path is not None:
        lines.append(f"# ontology: {ontology_path}")
    if ontology_hash_str is not None:
        lines.append(f"# ontology_hash: {ontology_hash_str}")
    (run_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def _write_run_meta(
    run_dir: Path,
    *,
    clip_ids: list[str],
    classes: list[str],
    train_ratio: float,
    per_clip_split: bool,
    seed: int,
    stats: ExportStats,
    ontology_hash_str: str | None,
) -> None:
    payload = {
        "run_id": run_dir.name,
        "clip_ids": clip_ids,
        "classes": classes,
        "train_ratio": train_ratio,
        "per_clip_split": per_clip_split,
        "seed": seed,
        "ontology_hash": ontology_hash_str,
        "train_frames": stats.train_frames,
        "val_frames": stats.val_frames,
        "train_polygons": stats.train_polygons,
        "val_polygons": stats.val_polygons,
        "skipped_frames": [
            {"clip_id": s.clip_id, "frame_idx": s.frame_idx, "reason": s.reason}
            for s in stats.skipped_frames
        ],
    }
    (run_dir / "export.json").write_text(json.dumps(payload, indent=2))


def _ontology_class_names(settings: Settings) -> tuple[list[str], str | None, str | None]:
    """Return (ordered class names, ontology_path, ontology_hash). Returns
    empty list when the ontology file does not exist; otherwise propagates
    parse errors so corrupt ontologies aren't silently masked by the
    manifest-derived fallback."""
    path = settings.resolve(settings.project.ontology)
    if not path.exists():
        return [], None, None
    mapping = load_ontology(path)
    seen: dict[str, None] = {}
    for v in mapping.values():
        seen.setdefault(v, None)
    return list(seen.keys()), str(path), ontology_hash(mapping)


def _classes_from_clips(clips: list[Clip]) -> list[str]:
    """Fallback class list (manifest-derived). class_id → label, sorted by id."""
    by_id: dict[int, str] = {}
    for c in clips:
        for t in c.tracks:
            by_id.setdefault(t.class_id, t.label)
    if not by_id:
        return []
    max_id = max(by_id)
    return [by_id.get(i, f"class_{i}") for i in range(max_id + 1)]


def export_run(
    store: ProjectStore,
    *,
    run_id: str,
    clip_ids: list[str] | None = None,
    train_ratio: float = 0.85,
    per_clip_split: bool = True,
    seed: int = 0,
    overwrite: bool = False,
    settings: Settings | None = None,
) -> ExportStats:
    """Materialize a YOLO-seg dataset under <runs_dir>/<run_id>/."""
    safe_id(run_id)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    all_clip_ids = store.list_clip_ids()
    if clip_ids is None:
        clip_ids = all_clip_ids
    else:
        missing = [c for c in clip_ids if c not in all_clip_ids]
        if missing:
            raise ValueError(f"unknown clip_ids: {missing}")
    if not clip_ids:
        raise ValueError("no clips to export")

    clips = [store.read_clip(cid) for cid in clip_ids]

    if settings is not None:
        classes, ont_path, ont_hash = _ontology_class_names(settings)
    else:
        classes, ont_path, ont_hash = [], None, None
    if not classes:
        classes = _classes_from_clips(clips)
    if not classes:
        raise ValueError("no classes resolvable from ontology or clip tracks")

    run_dir = store.run_dir(run_id)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run dir already exists: {run_dir}")
        # Belt-and-suspenders: assert the resolved path is strictly under
        # runs_dir before any rmtree.
        _assert_contained(run_dir, store.runs_dir)
        shutil.rmtree(run_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    frame_items: list[tuple[Clip, int]] = []
    stats = ExportStats(run_id=run_id, classes=list(classes))
    for clip in clips:
        deleted = set(clip.deleted_frames)
        labeled_frames: dict[int, list[Track]] = {}
        for t in clip.tracks:
            for fi in t.masks.keys():
                if fi in deleted:
                    continue
                labeled_frames.setdefault(fi, []).append(t)
        if not labeled_frames:
            stats.skipped_frames.append(SkippedFrame(clip.clip_id, -1, "no_labeled_frames"))
            continue
        for fi in sorted(labeled_frames.keys()):
            frame_items.append((clip, fi))

    if not frame_items:
        raise ValueError("no labeled frames found across requested clips")

    if per_clip_split:
        train_ids, val_ids = _split_per_clip(
            [c.clip_id for c in clips], train_ratio, rng
        )
        if not val_ids and len(clips) > 1:
            # Deterministic under the same seed; iterating a set isn't.
            picked = sorted(train_ids)[0]
            train_ids.remove(picked)
            val_ids.add(picked)
        train_keys: set = train_ids
    else:
        keys = [(c.clip_id, fi) for c, fi in frame_items]
        train_set, _val_set = _split_per_frame(keys, train_ratio, rng)
        train_keys = train_set

    def _is_train(clip: Clip, idx: int) -> bool:
        if per_clip_split:
            return clip.clip_id in train_keys
        return (clip.clip_id, idx) in train_keys

    for clip, fi in frame_items:
        split = "train" if _is_train(clip, fi) else "val"
        cdir = store.clip_dir(clip.clip_id)
        src_img = cdir / "frames" / f"{fi:06d}.jpg"
        if not src_img.exists():
            stats.skipped_frames.append(SkippedFrame(clip.clip_id, fi, "missing_jpg"))
            continue

        lines: list[str] = []
        polys_written = 0
        for t in clip.tracks:
            polys = _frame_polygons_for_track(t, fi, cdir)
            if not polys:
                continue
            for poly in polys:
                norm = _normalize_polygon(poly, clip.width, clip.height)
                lines.append(f"{t.class_id} " + " ".join(f"{v:.6f}" for v in norm))
                polys_written += 1
        if not lines:
            stats.skipped_frames.append(SkippedFrame(clip.clip_id, fi, "no_polygons"))
            continue

        stem = f"{clip.clip_id}__{fi:06d}"
        dst_img = run_dir / "images" / split / f"{stem}.jpg"
        dst_lbl = run_dir / "labels" / split / f"{stem}.txt"
        shutil.copyfile(src_img, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n")
        if split == "train":
            stats.train_frames += 1
            stats.train_polygons += polys_written
        else:
            stats.val_frames += 1
            stats.val_polygons += polys_written

    if stats.train_frames == 0 or stats.val_frames == 0:
        stats.skipped_frames.append(
            SkippedFrame(
                "__split__",
                -1,
                f"unbalanced train={stats.train_frames} val={stats.val_frames}",
            )
        )

    _write_data_yaml(
        run_dir,
        classes,
        ontology_path=ont_path,
        ontology_hash_str=ont_hash,
    )
    _write_run_meta(
        run_dir,
        clip_ids=clip_ids,
        classes=classes,
        train_ratio=train_ratio,
        per_clip_split=per_clip_split,
        seed=seed,
        stats=stats,
        ontology_hash_str=ont_hash,
    )
    return stats
