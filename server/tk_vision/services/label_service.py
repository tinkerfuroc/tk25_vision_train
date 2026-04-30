from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Awaitable, Callable

import cv2
import numpy as np

from ..annotate.chunker import chunk_frames
from ..annotate.sam3 import ChunkOOMError, Detection, Sam3Engine, is_degenerate_mask
from ..data.manifest import Clip, Mask, MaskSource, Track
from ..data.persistence import (
    ProjectStore,
    get_track,
    mask_path_for,
    next_track_id,
    read_mask,
    update_track,
    write_mask,
)
from ..config import Settings, load_ontology

log = logging.getLogger("tk_vision.label")

Point = tuple[int, int, int]
ProgressCb = Callable[[dict], Awaitable[None]]


def _write_masks_batch(items: list[tuple[Path, np.ndarray]]) -> None:
    for path, mask in items:
        write_mask(path, mask)


class LabelService:
    """Glue between the FastAPI label endpoints and the Sam3Engine.

    Owns no model state itself — just composes:
      * frame loading from disk (per ProjectStore)
      * inference through Sam3Engine (image mode for now)
      * RLE persistence + manifest updates
    """

    def __init__(
        self,
        store: ProjectStore,
        sam3: Sam3Engine,
        settings: Settings,
    ) -> None:
        self.store = store
        self.sam3 = sam3
        self.settings = settings

    # ------------------------------------------------------------------ frames
    def _read_frame(self, clip_id: str, idx: int) -> np.ndarray:
        p = self.store.frames_dir(clip_id) / f"{idx:06d}.jpg"
        if not p.is_file():
            raise FileNotFoundError(p)
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to decode {p}")
        return img

    # ------------------------------------------------------------------ seed
    async def seed_first_frame(self, clip_id: str) -> Clip:
        clip = self.store.read_clip(clip_id)
        ontology = load_ontology(self.settings.resolve(self.settings.project.ontology))
        prompts = list(ontology.keys())
        labels = list(ontology.values())

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, self._read_frame, clip_id, 0)
        frame_key = (clip_id, 0)
        detections = await loop.run_in_executor(
            None,
            lambda: self.sam3.predict_text(image, prompts, frame_key=frame_key),
        )
        log.info("seed: %s text-prompt detections", len(detections))

        accepted: list[tuple[Detection, int, str]] = []
        for det in detections:
            prompt = det.phrase or ""
            if prompt not in ontology:
                continue
            class_name = ontology[prompt]
            class_id = labels.index(class_name)
            accepted.append((det, class_id, class_name))

        # Wipe any prior "text" tracks so re-seeding is idempotent — both
        # in the manifest and on disk. Otherwise orphaned mask files survive
        # under masks/<old_tid>/ and bleed through to the SPA.
        cdir = self.store.clip_dir(clip_id)
        for t in clip.tracks:
            if t.seeded_from == "text":
                self._drop_track_disk(cdir, t.track_id)
        clip.tracks = [
            t for t in clip.tracks if t.seeded_from != "text"
        ]
        for det, class_id, label in accepted:
            tid = next_track_id(clip)
            track = Track(
                track_id=tid,
                class_id=class_id,
                label=label,
                seeded_from="text",
            )
            mask_p = mask_path_for(cdir, tid, 0)
            await loop.run_in_executor(None, write_mask, mask_p, det.mask)
            track.masks[0] = Mask(
                frame_idx=0,
                track_id=tid,
                rle_path=str(mask_p.relative_to(cdir)),
                source="text",
            )
            update_track(clip, track)

        self.store.write_clip(clip)
        return clip

    # ------------------------------------------------------------------ refine
    async def refine_frame(
        self,
        clip_id: str,
        frame_idx: int,
        *,
        track_id: int | None,
        points: list[Point] | None = None,
        box: tuple[int, int, int, int] | None = None,
        class_id: int | None = None,
        label: str | None = None,
    ) -> tuple[Clip, Track, Mask]:
        clip = self.store.read_clip(clip_id)
        ontology = load_ontology(self.settings.resolve(self.settings.project.ontology))
        labels = list(ontology.values())

        target: Track | None = None
        ref_mask: np.ndarray | None = None
        if track_id is not None:
            target = get_track(clip, track_id)
            if target is None:
                raise KeyError(f"Track {track_id} not found in clip {clip_id}")
            # Load existing mask if present on this frame — enables negative-only refinement
            if frame_idx in target.masks:
                try:
                    ref_mask = read_mask(mask_path_for(self.store.clip_dir(clip_id), track_id, frame_idx))
                    log.debug("refine_frame: loaded ref_mask for track %d frame %d", track_id, frame_idx)
                except FileNotFoundError:
                    log.warning("refine_frame: mask file missing for track %d frame %d", track_id, frame_idx)
        else:
            if class_id is None and label is None:
                raise ValueError("New track requires class_id or label")
            if class_id is None:
                class_id = labels.index(label)
            if label is None:
                label = labels[class_id]
            target = Track(
                track_id=next_track_id(clip),
                class_id=class_id,
                label=label,
                seeded_from="click",
            )

        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, self._read_frame, clip_id, frame_idx)
        frame_key = (clip_id, frame_idx)
        cdir = self.store.clip_dir(clip_id)
        prompt = target.label or ""

        if points:
            # Determine if this is a negative-only refinement
            positive_pts = [p for p in points if p[2] == 1]
            negative_pts = [p for p in points if p[2] == 0]
            if not positive_pts and negative_pts and ref_mask is None:
                raise ValueError(
                    "Negative-only refinement requires an existing mask. "
                    "Click on the object first (positive point), or select a track that has a mask on this frame."
                )
            det = await loop.run_in_executor(
                None,
                lambda: self.sam3.predict_clicks(
                    image, points=points, prompt=prompt, ref_mask=ref_mask, frame_key=frame_key
                ),
            )
        elif box is not None:
            det = await loop.run_in_executor(
                None,
                lambda: self.sam3.predict_box(
                    image, box, prompt=prompt, frame_key=frame_key
                ),
            )
        else:
            raise ValueError("refine_frame requires `points` or `box`")
        if det is None:
            if points and not positive_pts:
                raise ValueError(
                    "SAM3 could not generate a mask from negative points. "
                    "The reference mask may have been too small or the negative regions covered too much of it."
                )
            raise ValueError(
                f"SAM3 returned no segmentation for this prompt. "
                f"Try clicking closer to the center of the object, or use a different prompt."
            )

        mask_p = mask_path_for(cdir, target.track_id, frame_idx)
        await loop.run_in_executor(None, write_mask, mask_p, det.mask)
        source: MaskSource = "click" if track_id is None else "edited"
        mask_record = Mask(
            frame_idx=frame_idx,
            track_id=target.track_id,
            rle_path=str(mask_p.relative_to(cdir)),
            source=source,
            point_prompts=points or [],
        )
        target.masks[frame_idx] = mask_record
        update_track(clip, target)
        self.store.write_clip(clip)
        return clip, target, mask_record

    # ------------------------------------------------------------------ tracks CRUD
    @staticmethod
    def _drop_track_disk(clip_dir: Path, track_id: int) -> None:
        d = clip_dir / "masks" / str(track_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def delete_track(self, clip_id: str, track_id: int) -> Clip:
        clip = self.store.read_clip(clip_id)
        clip.tracks = [t for t in clip.tracks if t.track_id != track_id]
        self._drop_track_disk(self.store.clip_dir(clip_id), track_id)
        self.store.write_clip(clip)
        return clip

    def cleanup_clip(self, clip_id: str) -> tuple[Clip, list[int]]:
        return cleanup_clip(self.store, clip_id)

    def update_track(
        self,
        clip_id: str,
        track_id: int,
        *,
        class_id: int | None = None,
        label: str | None = None,
    ) -> Clip:
        ontology = load_ontology(self.settings.resolve(self.settings.project.ontology))
        labels = list(ontology.values())
        clip = self.store.read_clip(clip_id)
        track = get_track(clip, track_id)
        if track is None:
            raise KeyError(f"Track {track_id} not found")
        if class_id is None and label is not None:
            class_id = labels.index(label)
        if label is None and class_id is not None:
            label = labels[class_id]
        if class_id is not None:
            track.class_id = int(class_id)
        if label is not None:
            track.label = label
        update_track(clip, track)
        self.store.write_clip(clip)
        return clip

    def delete_frame(self, clip_id: str, frame_idx: int) -> Clip:
        clip = self.store.read_clip(clip_id)
        if frame_idx in clip.deleted_frames:
            return clip
        clip.deleted_frames.append(frame_idx)
        self.store.write_clip(clip)
        return clip

    def restore_frame(self, clip_id: str, frame_idx: int) -> Clip:
        clip = self.store.read_clip(clip_id)
        if frame_idx not in clip.deleted_frames:
            return clip
        clip.deleted_frames = [f for f in clip.deleted_frames if f != frame_idx]
        self.store.write_clip(clip)
        return clip

    def prune_frames(self, clip_id: str, from_idx: int) -> Clip:
        """Mark every frame in [from_idx, frame_count) as deleted."""
        clip = self.store.read_clip(clip_id)
        if from_idx < 0 or from_idx >= clip.frame_count:
            raise ValueError(
                f"from_idx {from_idx} out of range [0, {clip.frame_count})"
            )
        clip.deleted_frames = sorted(
            set(clip.deleted_frames) | set(range(from_idx, clip.frame_count))
        )
        self.store.write_clip(clip)
        return clip

    # ------------------------------------------------------------------ propagate
    async def propagate_clip(
        self,
        clip_id: str,
        *,
        start: int,
        end: int,
        chunk_size: int,
        chunk_overlap: int,
        respect_edits: bool,
        progress_cb: ProgressCb | None = None,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Clip:
        """Propagate existing tracks across [start, end) using the native
        SAM3 video tracker (`Sam3Engine.propagate_video`).

        Per chunk: collect each track's anchor mask (most recent mask at or
        before chunk.start, or the earliest if none exists upstream), seed
        the tracker with those, then iterate. Frames in
        ``[chunk.start, chunk.core_start)`` are written as
        ``source='propagated_overlap'`` and skipped on YOLO export.
        """
        clip = self.store.read_clip(clip_id)
        if not clip.tracks:
            raise ValueError("clip has no tracks; seed first")
        if start < 0 or end > clip.frame_count or end <= start:
            raise ValueError(f"invalid range [{start}, {end}) for {clip.frame_count} frames")

        chunks = chunk_frames(start=start, end=end, size=chunk_size, overlap=chunk_overlap)
        cdir = self.store.clip_dir(clip_id)
        loop = asyncio.get_running_loop()
        total_frames = end - start
        done_frames = 0

        for chunk in chunks:
            if cancel_cb and cancel_cb():
                log.info("propagate_clip(%s): cancelled before chunk %d", clip_id, chunk.start)
                return clip

            # Re-read clip in case prior chunk wrote masks via concurrent edits.
            clip = self.store.read_clip(clip_id)
            seed_masks: dict[int, np.ndarray] = {}
            for track in clip.tracks:
                anchor = self._best_anchor_frame(track, chunk.start + 1)
                if anchor is None:
                    continue
                try:
                    seed_masks[track.track_id] = read_mask(
                        mask_path_for(cdir, track.track_id, anchor)
                    )
                except FileNotFoundError:
                    continue
            if not seed_masks:
                if progress_cb:
                    await progress_cb(
                        {"event": "chunk_skipped", "chunk_start": chunk.start, "reason": "no_seed_masks"}
                    )
                continue

            try:
                chunk_imgs = [
                    self._read_frame(clip_id, fr) for fr in range(chunk.start, chunk.end)
                ]
            except FileNotFoundError as e:
                log.warning("propagate_clip: missing frame jpeg: %s", e)
                if progress_cb:
                    await progress_cb({"event": "chunk_skipped", "chunk_start": chunk.start, "reason": "missing_jpeg"})
                continue

            # Mutate the in-memory `clip` per yielded frame, persist once at
            # chunk end. Saves 50× full-manifest reads/writes per chunk.
            # Mask PNG writes for one frame are batched into a single
            # executor call to cut context-switch overhead.
            try:
                async for offset, masks in self._aiter_propagate(
                    chunk_imgs, seed_masks, cancel_cb, loop
                ):
                    absolute_fr = chunk.start + offset
                    if absolute_fr in clip.deleted_frames:
                        continue
                    source: MaskSource = (
                        "propagated" if absolute_fr >= chunk.core_start else "propagated_overlap"
                    )
                    pending_writes: list[tuple[Path, np.ndarray]] = []
                    for tid, mask in masks.items():
                        track = get_track(clip, tid)
                        if track is None:
                            continue
                        if respect_edits and absolute_fr in track.masks and track.masks[absolute_fr].source == "edited":
                            continue
                        p = mask_path_for(cdir, tid, absolute_fr)
                        pending_writes.append((p, mask))
                        track.masks[absolute_fr] = Mask(
                            frame_idx=absolute_fr,
                            track_id=tid,
                            rle_path=str(p.relative_to(cdir)),
                            source=source,
                        )
                        update_track(clip, track)
                    if pending_writes:
                        await loop.run_in_executor(None, _write_masks_batch, pending_writes)
                    done_frames += 1
                    if progress_cb:
                        await progress_cb(
                            {
                                "event": "frame",
                                "frame_idx": absolute_fr,
                                "done": done_frames,
                                "total": total_frames,
                                "n_tracks_in_frame": len(masks),
                                "source": source,
                            }
                        )
            except ChunkOOMError as e:
                log.error("propagate_clip: CUDA OOM at chunk start=%d: %s", chunk.start, e)
                raise RuntimeError(
                    f"CUDA out of memory during propagation at frame {chunk.start}. "
                    f"Try reducing chunk_size (current: {chunk_size}) in the propagate request."
                ) from e
            except Exception as e:  # noqa: BLE001
                log.exception("propagate_clip chunk failed at start=%d: %s", chunk.start, e)
                raise
            self.store.write_clip(clip)
            if progress_cb:
                await progress_cb({"event": "chunk_done", "chunk_end": chunk.end})

        return clip

    async def _aiter_propagate(
        self,
        frames: list[np.ndarray],
        seed_masks: dict[int, np.ndarray],
        cancel_cb: Callable[[], bool] | None,
        loop: asyncio.AbstractEventLoop,
    ):
        """Wrap `Sam3Engine.propagate_video` (a synchronous generator) into
        an async iterator. Each `next()` runs in the default executor so the
        event loop continues processing cancel signals."""
        log.info("_aiter_propagate: starting with %d frames, %d seed masks", len(frames), len(seed_masks))

        # Run the entire propagate_video in executor to avoid blocking
        def run_propagate():
            try:
                gen = self.sam3.propagate_video(
                    frames=frames, seed_masks=seed_masks, cancel_cb=cancel_cb
                )
                results = []
                for offset, masks in gen:
                    results.append((offset, masks))
                    if cancel_cb and cancel_cb():
                        break
                return results
            except Exception as e:
                log.exception("run_propagate error: %s", e)
                raise

        try:
            results = await loop.run_in_executor(None, run_propagate)
            log.info("_aiter_propagate: got %d results", len(results))
            for offset, masks in results:
                yield offset, masks
        except Exception as e:
            log.error("_aiter_propagate: error: %s", e)
            raise

    @staticmethod
    def _best_anchor_frame(track: Track, frame_idx: int) -> int | None:
        """Pick the most recent mask frame strictly before frame_idx; if
        none, fall back to the earliest mask (typically frame 0)."""
        prior = [f for f in track.masks if f < frame_idx]
        if prior:
            return max(prior)
        if track.masks:
            return min(track.masks.keys())
        return None


def cleanup_clip(store: ProjectStore, clip_id: str) -> tuple[Clip, list[int]]:
    """Drop tracks whose persisted masks are degenerate.

    A track is degenerate if any of its masks fails ``is_degenerate_mask``
    (full-frame, edge-touching, or below MIN_AREA_PX). User-edited tracks
    (``seeded_from=='click'``) are spared regardless — the user owns those.

    Standalone (no Sam3Engine instance required) so it can run without GPU
    via the cleanup CLI on existing project state.
    """
    clip = store.read_clip(clip_id)
    cdir = store.clip_dir(clip_id)
    bad_tids = set(scan_degenerate_track_ids(store, clip))
    dropped = sorted(bad_tids)
    for tid in dropped:
        d = cdir / "masks" / str(tid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    clip.tracks = [t for t in clip.tracks if t.track_id not in bad_tids]
    store.write_clip(clip)
    log.info("cleanup_clip(%s): dropped %d tracks: %s", clip_id, len(dropped), dropped)
    return clip, dropped


def scan_degenerate_track_ids(store: ProjectStore, clip: Clip) -> list[int]:
    """Return track_ids whose persisted masks fail `is_degenerate_mask`.
    Click-seeded tracks are spared regardless."""
    cdir = store.clip_dir(clip.clip_id)
    out: list[int] = []
    for track in clip.tracks:
        if track.seeded_from == "click":
            continue
        bad = False
        for fr_idx in track.masks:
            p = mask_path_for(cdir, track.track_id, fr_idx)
            if not p.is_file():
                continue
            try:
                mask = read_mask(p)
            except Exception:
                bad = True
                break
            if is_degenerate_mask(
                mask,
                max_area_frac=Sam3Engine.MAX_MASK_AREA_FRAC,
                min_area_px=Sam3Engine.MIN_MASK_AREA_PX,
            ):
                bad = True
                break
        if bad:
            out.append(track.track_id)
    return out
