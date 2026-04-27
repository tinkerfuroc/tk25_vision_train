"""Augmentation pass for an exported YOLO-seg run.

Reads `<run_dir>/images/train/*.jpg` + `<run_dir>/labels/train/*.txt`, applies
Albumentations transforms `multiplier - 1` extra times per source image (the
original is renamed to `<stem>_aug0.jpg`/`.txt`), and writes augmented
copies alongside.

YOLO-seg label format: `class x0 y0 x1 y1 ... xn yn` normalized to (0,1).
We reconstruct masks per-polygon, transform together with the image via
`A.Compose(masks=...)`, then re-extract polygons (multi-contour aware).

Copy-paste augmentation: with prob `p`, paste a random labeled instance from
a pool (train split) onto the destination image, masking the destination
beneath the pasted instance. The pasted polygon is added as an extra row.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import AugmentCfg
from .polygons import mask_to_polygons, normalize_polygon, polygon_to_mask

_AUG_SUFFIX_RE = re.compile(r"_aug\d+$")


def _load_label_file(path: Path) -> list[tuple[int, list[float]]]:
    out: list[tuple[int, list[float]]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        out.append((cls, coords))
    return out


# Re-exports — tests + callers use the bare names.
_mask_to_polygons = mask_to_polygons
_polygon_to_mask = polygon_to_mask
_normalize = normalize_polygon


def _build_pipeline(cfg: AugmentCfg):
    """Translate the YAML ops list into an `A.Compose`.

    Ops are decoded permissively: unknown keys passed straight through to the
    Albumentations constructor, unknown op names are skipped.
    """
    import albumentations as A  # local: keep import cost out of API hot paths.

    name_map = {
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "HueSaturationValue": A.HueSaturationValue,
        "MotionBlur": A.MotionBlur,
        "GaussNoise": A.GaussNoise,
        "Affine": A.Affine,
        "CoarseDropout": A.CoarseDropout,
        "RandomScale": A.RandomScale,
        "Rotate": A.Rotate,
    }
    transforms = []
    for spec in cfg.ops:
        if not isinstance(spec, dict) or "name" not in spec:
            continue
        cls = name_map.get(spec["name"])
        if cls is None:
            continue
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        for k, v in list(kwargs.items()):
            if isinstance(v, list) and len(v) == 2:
                kwargs[k] = tuple(v)
        try:
            transforms.append(cls(**kwargs))
        except TypeError:
            transforms.append(cls(p=kwargs.get("p", 0.5)))
    return A.Compose(transforms)


@dataclass
class AugmentStats:
    run_id: str
    source_frames: int = 0
    written_frames: int = 0
    written_polygons: int = 0
    copy_paste_inserts: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _PoolEntry:
    image_path: Path
    cls: int
    coords: list[float]


def _gather_pool(
    run_dir: Path, split: str, *, limit: int, rng: random.Random
) -> list[_PoolEntry]:
    """Reservoir-sampled pool of `_PoolEntry` rows from `<split>` labels."""
    entries: list[_PoolEntry] = []
    images = sorted((run_dir / "images" / split).glob("*.jpg"))
    n = 0
    for img in images:
        lbl = run_dir / "labels" / split / (img.stem + ".txt")
        if not lbl.exists():
            continue
        for cls, coords in _load_label_file(lbl):
            n += 1
            entry = _PoolEntry(img, cls, coords)
            if len(entries) < limit:
                entries.append(entry)
            else:
                j = rng.randrange(n)
                if j < limit:
                    entries[j] = entry
    return entries


class _DonorImageCache:
    """Tiny LRU for decoded donor JPEGs. Bound stops a long copy-paste run from
    pinning every train image in RAM."""

    def __init__(self, max_entries: int = 32) -> None:
        self.max = max_entries
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray | None:
        img = self._cache.get(path)
        if img is not None:
            self._cache.move_to_end(path)
            return img
        loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if loaded is None:
            return None
        self._cache[path] = loaded
        if len(self._cache) > self.max:
            self._cache.popitem(last=False)
        return loaded


def _do_copy_paste(
    image: np.ndarray,
    masks: list[np.ndarray],
    classes: list[int],
    pool: list[_PoolEntry],
    cache: _DonorImageCache,
    *,
    rng: random.Random,
    min_area_frac: float,
) -> bool:
    """Mutate image/masks/classes in place by pasting one donor."""
    if not pool:
        return False
    donor = rng.choice(pool)
    donor_img = cache.get(donor.image_path)
    if donor_img is None:
        return False
    dh, dw = donor_img.shape[:2]
    donor_mask = polygon_to_mask(donor.coords, dh, dw)
    h, w = image.shape[:2]
    if donor_mask.sum() < min_area_frac * h * w:
        return False
    if (dh, dw) != (h, w):
        donor_img = cv2.resize(donor_img, (w, h), interpolation=cv2.INTER_LINEAR)
        donor_mask = cv2.resize(donor_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    region = donor_mask.astype(bool)
    image[region] = donor_img[region]
    for m in masks:
        m[region] = 0
    masks.append(donor_mask)
    classes.append(donor.cls)
    return True


def augment_run(
    run_dir: Path,
    cfg: AugmentCfg,
    *,
    seed: int = 0,
    splits: tuple[str, ...] = ("train",),
) -> AugmentStats:
    """Augment images in `run_dir` per `cfg`. Mutates the run dir in place."""
    if cfg.multiplier <= 1:
        raise ValueError(f"multiplier must be > 1, got {cfg.multiplier}")
    if not (run_dir / "images" / "train").exists():
        raise FileNotFoundError(f"missing {run_dir}/images/train; run export first")

    pipeline = _build_pipeline(cfg)
    rng = random.Random(seed)
    stats = AugmentStats(run_id=run_dir.name)

    cp = cfg.copy_paste
    pool: list[_PoolEntry] = []
    if cp.enabled:
        pool = _gather_pool(run_dir, "train", limit=cp.pool_size, rng=rng)
    donor_cache = _DonorImageCache()

    if cfg.apply_to == "all":
        targets = list(splits)
    else:
        targets = [s for s in splits if s == "train"]

    for split in targets:
        img_dir = run_dir / "images" / split
        lbl_dir = run_dir / "labels" / split
        for src_img in sorted(img_dir.glob("*.jpg")):
            if _AUG_SUFFIX_RE.search(src_img.stem):
                continue
            src_lbl = lbl_dir / (src_img.stem + ".txt")
            if not src_lbl.exists():
                stats.skipped.append((src_img.name, "missing_label"))
                continue
            image_bgr = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
            if image_bgr is None:
                stats.skipped.append((src_img.name, "imread_failed"))
                continue
            h, w = image_bgr.shape[:2]
            rows = _load_label_file(src_lbl)
            if not rows:
                stats.skipped.append((src_img.name, "no_polygons"))
                continue
            stats.source_frames += 1
            base_classes = [cls for cls, _ in rows]
            base_masks = [polygon_to_mask(coords, h, w) for _, coords in rows]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # variant 0 is the original — rename it instead of re-encoding.
            stem0 = f"{src_img.stem}_aug0"
            shutil.move(str(src_img), str(img_dir / f"{stem0}.jpg"))
            shutil.move(str(src_lbl), str(lbl_dir / f"{stem0}.txt"))
            stats.written_frames += 1
            stats.written_polygons += len(rows)

            for variant in range(1, cfg.multiplier):
                aug = pipeline(image=image_rgb, masks=base_masks)
                out_rgb = aug["image"]
                out_img = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                out_masks = [np.asarray(m, dtype=np.uint8) for m in aug["masks"]]
                out_classes = list(base_classes)
                if cp.enabled and rng.random() < cp.p:
                    if _do_copy_paste(
                        out_img,
                        out_masks,
                        out_classes,
                        pool,
                        donor_cache,
                        rng=rng,
                        min_area_frac=cp.min_area_frac,
                    ):
                        stats.copy_paste_inserts += 1

                lines: list[str] = []
                polys_written = 0
                for cls, mask in zip(out_classes, out_masks):
                    polys = mask_to_polygons(mask)
                    for p in polys:
                        norm = normalize_polygon(p, w, h)
                        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
                        polys_written += 1
                if not lines:
                    continue
                stem = f"{src_img.stem}_aug{variant}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), out_img)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                stats.written_frames += 1
                stats.written_polygons += polys_written

    (run_dir / "augment.json").write_text(
        json.dumps(
            {
                "run_id": stats.run_id,
                "source_frames": stats.source_frames,
                "written_frames": stats.written_frames,
                "written_polygons": stats.written_polygons,
                "copy_paste_inserts": stats.copy_paste_inserts,
                "multiplier": cfg.multiplier,
                "seed": seed,
                "skipped": [{"file": f, "reason": r} for f, r in stats.skipped],
            },
            indent=2,
        )
    )
    return stats
