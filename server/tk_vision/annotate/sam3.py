from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, Callable

import numpy as np
import torch
from scipy.ndimage import binary_dilation

log = logging.getLogger("tk_vision.sam3")


@dataclass
class Detection:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    phrase: str | None = None


class ChunkOOMError(RuntimeError):
    """Raised when the SAM3 video tracker exhausts CUDA memory mid-chunk.
    Caller is expected to halve the chunk size and retry."""


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unsupported sam3 dtype {name!r}")
    return _DTYPES[name]


def _xyxy_to_cxcywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


_FRAME_CACHE_CAPACITY = 4


def is_degenerate_mask(mask: np.ndarray, *, max_area_frac: float, min_area_px: int) -> bool:
    """Return True if `mask` should be rejected as detector hallucination."""
    if mask.ndim != 2:
        return True
    h, w = mask.shape
    area = int(mask.sum())
    if area < min_area_px:
        return True
    if area > int(max_area_frac * h * w):
        return True
    if (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    ):
        return True
    return False


class Sam3Engine:
    """Thin wrapper around HuggingFace's Sam3VideoModel.

    All public methods serialize through a single threading.Lock so concurrent
    HTTP requests are safe (sub-second per call → serializing is fine). Vision
    features are cached by `frame_key` (typically `(clip_id, frame_idx)`) so
    seed-then-clicks on the same frame skips the 1008² encoder pass.
    """

    # Sigmoid-probability cutoff for binarizing mask logits. SAM3 query masks
    # smear past the object on low-confidence detections so 0.5 (vs 0.0) is
    # essential.
    MASK_THRESHOLD = 0.5
    MAX_DETECTIONS_PER_PROMPT = 8
    # Reject hallucinated frame-covering masks.
    MAX_MASK_AREA_FRAC = 0.45
    MIN_MASK_AREA_PX = 64

    ScoreMode = Literal["native", "per_query"]

    def __init__(
        self,
        model_dir: str | Path = "./sam3_checkpoint_hf",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        text_threshold: float = 0.3,
        box_threshold: float = 0.4,
        score_mode: "Sam3Engine.ScoreMode" = "native",
    ) -> None:
        self.model_dir = str(Path(model_dir).resolve())
        self.device = device
        self.dtype = dtype
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.text_threshold = float(text_threshold)
        self.box_threshold = float(box_threshold)
        self.score_mode: "Sam3Engine.ScoreMode" = score_mode
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._detector = None
        self._vision_cache: "collections.OrderedDict[tuple, object]" = collections.OrderedDict()
        log.info("Sam3Engine: initialized; not yet loaded")

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from transformers import Sam3Processor, Sam3VideoModel

            log.info("Sam3Engine: loading Sam3Processor from %s", self.model_dir)
            self._processor = Sam3Processor.from_pretrained(self.model_dir)
            log.info("Sam3Engine: loading Sam3VideoModel (%s) on %s", self.dtype, self.device)
            model = Sam3VideoModel.from_pretrained(self.model_dir, dtype=self.torch_dtype)
            patched = self._patch_tracker_neck(model)
            if patched != 22:
                raise RuntimeError(
                    f"Sam3Engine: tracker_neck patch loaded {patched}/22 weights from "
                    f"{self.model_dir}/model.safetensors. Cannot start native SAM3 "
                    f"video tracking without these. Install a complete checkpoint with "
                    f"`tk_vision fetch-weights --source hf --repo <repo> --verify --yes`."
                )
            log.info("Sam3Engine: patched %d/22 tracker_neck weights", patched)
            model = model.to(self.device).eval()
            self._model = model
            self._detector = model.detector_model
            if torch.cuda.is_available():
                log.info(
                    "Sam3Engine: loaded; vram=%.2f GB",
                    torch.cuda.memory_allocated() / 1e9,
                )

    @staticmethod
    def _patch_tracker_neck(model) -> int:
        """Alias `tracker_model.tracker_neck.*` from the on-disk safetensors
        to top-level `tracker_neck.*` in the live model.

        HF's auto-loader can't disambiguate this suffix collision (two ckpt
        keys share the same suffix `tracker_neck.fpn_layers.<i>.<x>`) and
        silently drops both — leaving the tracker FPN at random init. We
        resolve it by hand. Returns the number of parameters successfully
        copied; caller must verify the count == 22.
        """
        from pathlib import Path
        from safetensors.torch import safe_open

        ckpt_path = Path(model.config.name_or_path) / "model.safetensors"
        if not ckpt_path.is_file():
            log.warning("Sam3Engine: %s missing; skipping tracker_neck patch", ckpt_path)
            return 0
        neck = model.tracker_neck
        target_dtype = next(neck.parameters()).dtype
        loaded = 0
        with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name, param in neck.named_parameters():
                src = f"tracker_model.tracker_neck.{name}"
                if src not in available:
                    continue
                tensor = f.get_tensor(src).to(target_dtype)
                if tensor.shape != param.shape:
                    log.warning(
                        "Sam3Engine: tracker_neck shape mismatch on %s (%s vs %s)",
                        name, tensor.shape, param.shape,
                    )
                    continue
                with torch.no_grad():
                    param.copy_(tensor)
                loaded += 1
        return loaded

    def is_loaded(self) -> bool:
        return self._model is not None

    def release(self) -> None:
        with self._lock:
            self._vision_cache.clear()
            self._detector = None
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def invalidate_frame(self, frame_key: tuple) -> None:
        with self._lock:
            self._vision_cache.pop(frame_key, None)

    def predict_text(
        self,
        image_bgr: np.ndarray,
        prompts: list[str],
        *,
        score_threshold: Optional[float] = None,
        max_per_prompt: Optional[int] = None,
        frame_key: tuple | None = None,
    ) -> list[Detection]:
        """Open-vocab segmentation: one forward per prompt."""
        if not prompts:
            return []
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        thr = self.text_threshold if score_threshold is None else float(score_threshold)
        topk = self.MAX_DETECTIONS_PER_PROMPT if max_per_prompt is None else max_per_prompt

        with self._lock:
            with torch.inference_mode(), torch.autocast(self.device, dtype=self.torch_dtype):
                vision_embeds = self._get_vision_embeds_locked(rgb, frame_key)
                results: list[Detection] = []
                for prompt in prompts:
                    out = self._detector(
                        vision_embeds=vision_embeds,
                        **self._tokenize(prompt),
                    )
                    results.extend(
                        self._decode_with_processor(out, (h, w), prompt, thr, topk)
                    )
        return results

    def predict_box(
        self,
        image_bgr: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        prompt: str = "",
        *,
        score_threshold: Optional[float] = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        """Box-prompted segmentation."""
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        thr = self.text_threshold if score_threshold is None else float(score_threshold)
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        cx, cy, bw, bh = _xyxy_to_cxcywh(x1, y1, x2, y2)
        cx /= w; bw /= w; cy /= h; bh /= h

        with self._lock:
            with torch.inference_mode(), torch.autocast(self.device, dtype=self.torch_dtype):
                vision_embeds = self._get_vision_embeds_locked(rgb, frame_key)
                box_t = torch.tensor([[[cx, cy, bw, bh]]], dtype=torch.float32, device=self.device)
                box_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)
                out = self._detector(
                    vision_embeds=vision_embeds,
                    input_boxes=box_t,
                    input_boxes_labels=box_labels,
                    **self._tokenize(prompt or "."),
                )
        candidates = self._decode_with_processor(out, (h, w), prompt or None, thr, 1)
        return candidates[0] if candidates else None

    def predict_clicks(
        self,
        image_bgr: np.ndarray,
        *,
        points: list[tuple[int, int, int]],
        prompt: str = "",
        ref_mask: np.ndarray | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        """Click-prompted refinement.

        If positive points exist: collapse them into a bounding box (with 12px
        padding) and route through predict_box.

        If only negative points: require ref_mask (existing mask to refine).
        Compute box from ref_mask, then apply mask subtraction after prediction.

        The prompt defaults to "." (generic) but should be the track's label
        when refining an existing track — helps SAM3 localize the object.
        """
        positive = [(x, y) for x, y, lbl in points if int(lbl) == 1]
        negative = [(x, y) for x, y, lbl in points if int(lbl) == 0]
        h, w = image_bgr.shape[:2]

        if not positive and not negative:
            return None

        # Case 1: Positive points exist — box around them
        if positive:
            xs = [int(x) for x, _ in positive]
            ys = [int(y) for _, y in positive]
            pad = 12
            box = (
                max(0, min(xs) - pad),
                max(0, min(ys) - pad),
                min(w, max(xs) + pad),
                min(h, max(ys) + pad),
            )
            det = self.predict_box(image_bgr, box, prompt=prompt or ".", frame_key=frame_key)
            if det is None:
                return None
            # Apply negative points: zero out mask regions under negative clicks
            if negative and det is not None:
                mask = det.mask.copy()
                neg_radius = 20  # pixels to zero out around each negative click
                for nx, ny in negative:
                    y_start = max(0, ny - neg_radius)
                    y_end = min(h, ny + neg_radius)
                    x_start = max(0, nx - neg_radius)
                    x_end = min(w, nx + neg_radius)
                    mask[y_start:y_end, x_start:x_end] = False
                # Reject if mask becomes too small after subtraction
                if mask.sum() < self.MIN_MASK_AREA_PX:
                    return None
                det = Detection(
                    mask=mask,
                    bbox=det.bbox,
                    score=det.score,
                    phrase=det.phrase,
                )
            return det

        # Case 2: Only negative points — need existing mask as reference
        if not positive and negative:
            if ref_mask is None:
                log.warning("predict_clicks: negative-only prompt requires ref_mask")
                return None
            # Compute bounding box from existing mask
            ys, xs = np.where(ref_mask)
            if len(xs) == 0 or len(ys) == 0:
                return None  # Empty reference mask
            pad = 16
            box = (
                max(0, int(xs.min()) - pad),
                max(0, int(ys.min()) - pad),
                min(w, int(xs.max()) + pad),
                min(h, int(ys.max()) + pad),
            )
            det = self.predict_box(image_bgr, box, prompt=prompt or ".", frame_key=frame_key)
            if det is None:
                return None
            # Apply negative points to exclude regions
            mask = det.mask.copy()
            neg_radius = 25  # larger radius for negative-only refinement
            for nx, ny in negative:
                y_start = max(0, ny - neg_radius)
                y_end = min(h, ny + neg_radius)
                x_start = max(0, nx - neg_radius)
                x_end = min(w, nx + neg_radius)
                mask[y_start:y_end, x_start:x_end] = False
            # Also exclude regions outside original ref_mask boundary
            # (keeps the mask roughly within the original object bounds)
            # Grow ref_mask slightly to allow some expansion, then intersect
            kernel = np.ones((15, 15), dtype=bool)
            expanded_ref = binary_dilation(ref_mask, structure=kernel)
            mask = mask & expanded_ref
            if mask.sum() < self.MIN_MASK_AREA_PX:
                return None
            # Recompute bbox from final mask
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
            return Detection(
                mask=mask,
                bbox=bbox,
                score=det.score,
                phrase=det.phrase,
            )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ------------------------------------------------------------------ video
    def propagate_video(
        self,
        *,
        frames: list[np.ndarray],
        seed_masks: dict[int, np.ndarray],
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Native SAM3 video tracking via the SAM2-style tracker.

        Yields ``(offset, {track_id: bool_mask})`` per frame.

        Implementation notes:
          * `Sam3VideoModel.tracker_model` is built with
            `remove_vision_encoder=True`, so we cannot call the tracker's
            vision_encoder. Instead we pre-compute per-frame features via
            `detector_model.get_vision_features` + the parent model's
            `get_vision_features_for_tracker` (which uses `tracker_neck`),
            then seed those into `session.cache.cache_vision_features`.
            The tracker reads from cache and skips the missing encoder.
          * Seed masks at frame 0 condition the tracker; SAM2 memory carries
            identity across subsequent frames.

        Raises ``ChunkOOMError`` on CUDA OOM.
        """
        try:
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                Sam3TrackerVideoInferenceSession,
            )
        except ImportError as e:
            log.error("propagate_video: failed to import Sam3TrackerVideoInferenceSession: %s", e)
            raise RuntimeError(
                "SAM3 video tracking requires transformers with Sam3TrackerVideo support. "
                "Ensure you have the correct version installed."
            ) from e

        if not frames:
            log.warning("propagate_video: no frames provided")
            return
        if not seed_masks:
            raise ValueError("seed_masks must be non-empty")

        log.info("propagate_video: starting with %d frames, %d seed masks", len(frames), len(seed_masks))
        self._ensure_loaded()

        h_full, w_full = frames[0].shape[:2]
        n_frames = len(frames)
        log.info("propagate_video: frame size %dx%d, %d frames", w_full, h_full, n_frames)

        # Preprocess all frames in one HF processor call → (T, 3, 1008, 1008).
        log.debug("propagate_video: preprocessing frames...")
        rgb_frames = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        try:
            inputs = self._processor(images=rgb_frames, return_tensors="pt")
        except Exception as e:
            log.error("propagate_video: processor failed: %s", e)
            raise
        video = inputs["pixel_values"]  # already (T, 3, 1008, 1008)
        log.debug("propagate_video: video tensor shape %s", video.shape)

        try:
            with self._lock:
                yield from self._propagate_locked(
                    Sam3TrackerVideoInferenceSession,
                    video,
                    seed_masks,
                    h_full,
                    w_full,
                    n_frames,
                    cancel_cb,
                )
        except torch.cuda.OutOfMemoryError as e:
            log.error("propagate_video: CUDA OOM: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise ChunkOOMError(str(e)) from e
        except Exception as e:
            log.exception("propagate_video: unexpected error: %s", e)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _propagate_locked(
        self,
        SessionCls,
        video: "torch.Tensor",
        seed_masks: dict[int, np.ndarray],
        h_full: int,
        w_full: int,
        n_frames: int,
        cancel_cb: Optional[Callable[[], bool]],
    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        log.info("_propagate_locked: creating session for %d frames", n_frames)
        with torch.inference_mode(), torch.autocast(
            self.device, dtype=self.torch_dtype
        ):
            try:
                session = SessionCls(
                    video=video,
                    video_height=h_full,
                    video_width=w_full,
                    inference_device=self.device,
                    inference_state_device="cpu",
                    video_storage_device=self.device,
                    dtype=self.torch_dtype,
                    # Cache all chunk frames so the tracker's per-frame lookup
                    # never falls through to the (removed) vision_encoder.
                    max_vision_features_cache_size=max(n_frames, 1),
                )
            except Exception as e:
                log.exception("_propagate_locked: failed to create session: %s", e)
                raise

            # Pre-compute vision features for every frame and seed the cache.
            log.info("_propagate_locked: computing vision features for %d frames", n_frames)
            for fr in range(n_frames):
                try:
                    px = session.get_frame(fr).unsqueeze(0)  # (1, 3, 1008, 1008)
                    vision_embeds = self._model.detector_model.get_vision_features(
                        pixel_values=px
                    )
                    feats, pos = self._model.get_vision_features_for_tracker(
                        vision_embeds=vision_embeds
                    )
                    session.cache.cache_vision_features(
                        fr, {"vision_feats": feats, "vision_pos_embeds": pos}
                    )
                    # Log GPU memory every 10 frames to help diagnose OOM
                    if fr % 10 == 9 and torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        log.debug("_propagate_locked: frame %d, GPU mem: %.2f GB allocated, %.2f GB reserved", fr + 1, allocated, reserved)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        log.error("_propagate_locked: CUDA OOM at frame %d/%d. Try reducing chunk_size.", fr, n_frames)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise ChunkOOMError(
                            f"CUDA OOM while computing vision features at frame {fr}/{n_frames}. "
                            f"Reduce chunk_size (current chunk has {n_frames} frames)."
                        ) from e
                    raise
                except Exception as e:
                    log.exception("_propagate_locked: failed at frame %d: %s", fr, e)
                    raise
            log.info("_propagate_locked: vision features cached")

            # Register each seeded track as an object with its anchor mask
            # at frame 0. Mask is upsampled internally by the tracker's
            # prompt encoder to mask_input_size, so we pass at native res.
            log.info("_propagate_locked: registering %d seed masks", len(seed_masks))
            for track_id, mask in seed_masks.items():
                if mask.shape != (h_full, w_full):
                    raise ValueError(
                        f"seed_mask for track {track_id} shape {mask.shape} "
                        f"does not match frames {(h_full, w_full)}"
                    )
                obj_idx = session.obj_id_to_idx(int(track_id))
                m_t = (
                    torch.from_numpy(mask.astype(np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device, dtype=self.torch_dtype)
                )
                session.add_mask_inputs(obj_idx, 0, m_t)
            session.obj_with_new_inputs = [int(t) for t in seed_masks.keys()]
            log.info("_propagate_locked: starting tracker iteration")

            tracker = self._model.tracker_model
            yield from self._iter_tracker(
                tracker, session, n_frames, (h_full, w_full), cancel_cb
            )

    def _iter_tracker(
        self,
        tracker,
        session,
        n_frames: int,
        target_size: tuple[int, int],
        cancel_cb: Optional[Callable[[], bool]],
    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        h_full, w_full = target_size

        try:
            for out in tracker.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=0,
                max_frame_num_to_track=n_frames,
            ):
                if cancel_cb is not None and cancel_cb():
                    break
                offset = int(out.frame_idx)
                masks_out: dict[int, np.ndarray] = {}
                pred = out.pred_masks  # (num_obj, 1, h_low, w_low) float
                obj_ids = out.obj_ids if hasattr(out, "obj_ids") else session.obj_ids
                if pred is None or len(pred) == 0:
                    yield offset, masks_out
                    continue
                full = torch.nn.functional.interpolate(
                    pred.to(torch.float32),
                    size=(h_full, w_full),
                    mode="bilinear",
                    align_corners=False,
                )  # (num_obj, 1, H, W)
                bin_masks = (full.sigmoid() > self.MASK_THRESHOLD).cpu().numpy()
                for i, obj_id in enumerate(obj_ids):
                    m = bin_masks[i, 0]
                    if is_degenerate_mask(
                        m, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX
                    ):
                        continue
                    masks_out[int(obj_id)] = m
                yield offset, masks_out
        except RuntimeError as e:
            # CUDA errors often manifest as RuntimeError
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                log.error("_iter_tracker: CUDA error: %s", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ChunkOOMError(str(e)) from e
            log.exception("_iter_tracker: runtime error: %s", e)
            raise
        except Exception as e:
            log.exception("_iter_tracker: unexpected error: %s", e)
            raise

    def _tokenize(self, text: str) -> dict[str, "torch.Tensor"]:
        enc = self._processor.tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=32, truncation=True
        )
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

    def _get_vision_embeds_locked(self, rgb_image: np.ndarray, frame_key: tuple | None):
        """Caller holds self._lock."""
        if frame_key is not None and frame_key in self._vision_cache:
            self._vision_cache.move_to_end(frame_key)
            log.debug("Sam3Engine: vision-cache hit for %s", frame_key)
            return self._vision_cache[frame_key]
        img_inputs = self._processor(images=rgb_image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.device)
        embeds = self._detector.get_vision_features(pixel_values=pixel_values)
        if frame_key is not None:
            self._vision_cache[frame_key] = embeds
            self._vision_cache.move_to_end(frame_key)
            while len(self._vision_cache) > _FRAME_CACHE_CAPACITY:
                self._vision_cache.popitem(last=False)
        return embeds

    def _decode_with_processor(
        self,
        out,
        target_size: tuple[int, int],
        phrase: str | None,
        score_threshold: float,
        max_keep: int,
    ) -> list[Detection]:
        if self.score_mode == "per_query":
            scores, boxes, masks = self._decode_per_query(out, target_size)
        else:
            scores, boxes, masks = self._decode_native(out, target_size, score_threshold)
        if scores.size == 0:
            return []

        h, w = target_size
        order = np.argsort(-scores)
        kept: list[Detection] = []
        for idx in order:
            score = float(scores[idx])
            if self.score_mode == "per_query" and score <= score_threshold:
                continue
            if len(kept) >= max_keep:
                break
            mask = masks[idx]
            if is_degenerate_mask(
                mask, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX
            ):
                continue
            if boxes is not None:
                x1, y1, x2, y2 = boxes[idx].tolist()
            else:
                ys, xs = np.where(mask)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
            kept.append(
                Detection(
                    mask=mask,
                    bbox=(
                        int(max(0, x1)),
                        int(max(0, y1)),
                        int(min(w, x2)),
                        int(min(h, y2)),
                    ),
                    score=score,
                    phrase=phrase,
                )
            )
        return kept

    def _decode_native(
        self, out, target_size: tuple[int, int], score_threshold: float
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        h, w = target_size
        results = self._processor.post_process_instance_segmentation(
            out,
            threshold=score_threshold,
            mask_threshold=self.MASK_THRESHOLD,
            target_sizes=[(h, w)],
        )
        if not results:
            return np.empty((0,), dtype=np.float32), None, np.empty((0, h, w), dtype=bool)
        r = results[0]
        if r.get("scores") is None or r.get("masks") is None or len(r["scores"]) == 0:
            return np.empty((0,), dtype=np.float32), None, np.empty((0, h, w), dtype=bool)
        scores = r["scores"].detach().to(torch.float32).cpu().numpy()
        masks = r["masks"].detach().cpu().numpy().astype(bool)
        boxes = (
            r["boxes"].detach().to(torch.float32).cpu().numpy()
            if r.get("boxes") is not None
            else None
        )
        return scores, boxes, masks

    def _decode_per_query(
        self, out, target_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score = sigmoid(pred_logits) only — ignore the presence head."""
        h, w = target_size
        scores_t = out.pred_logits[0].detach().to(torch.float32).sigmoid()
        boxes_t = out.pred_boxes[0].detach().to(torch.float32)
        masks_low = out.pred_masks[0].detach().to(torch.float32).sigmoid()
        # Scale normalized xyxy → pixel coords.
        scale = torch.tensor([w, h, w, h], dtype=boxes_t.dtype, device=boxes_t.device)
        boxes_t = boxes_t * scale
        masks_full = torch.nn.functional.interpolate(
            masks_low.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        masks_bin = masks_full > self.MASK_THRESHOLD
        return (
            scores_t.cpu().numpy(),
            boxes_t.cpu().numpy(),
            masks_bin.cpu().numpy(),
        )
