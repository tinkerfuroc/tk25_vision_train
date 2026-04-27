from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 28000
    bind_warning: bool = True
    cors_origins: list[str] = Field(default_factory=list)


class ProjectCfg(BaseModel):
    name: str = "tk25"
    data_root: str = "./data"
    ontology: str = "./resource/ontology.json"


class CaptureCfg(BaseModel):
    source: str = "live"
    bag_path: str | None = None
    folder_path: str | None = None
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    max_clip_seconds: int = 30


class Sam3Cfg(BaseModel):
    model_dir: str = "./sam3_checkpoint"
    device: str = "cuda"
    dtype: str = "bfloat16"
    text_threshold: float = 0.3
    box_threshold: float = 0.4
    # `native`     = HF post-process: score = sigmoid(logits) * sigmoid(presence)
    # `per_query`  = ignore presence head; score = sigmoid(logits) only.
    # Use `per_query` when ontology prompts are out of SAM3's training
    # distribution (long phrases, domain jargon) and the presence head
    # collapses scores below threshold.
    score_mode: Literal["native", "per_query"] = "native"


class PropagateCfg(BaseModel):
    chunk_size: int = 50
    chunk_overlap: int = 4
    respect_edits: bool = True


class LabelCfg(BaseModel):
    fill_hole_area: int = 16
    min_area_px: int = 50
    poly_epsilon: float = 1.5


class CopyPasteCfg(BaseModel):
    enabled: bool = False
    p: float = 0.0
    pool: str = "train_only"
    min_area_frac: float = 0.005
    pool_size: int = 256


class AugmentCfg(BaseModel):
    multiplier: int = 4
    apply_to: Literal["train_only", "all"] = "train_only"
    ops: list[dict[str, Any]] = Field(default_factory=list)
    copy_paste: CopyPasteCfg = Field(default_factory=CopyPasteCfg)


class TrainCfg(BaseModel):
    base_weights: str = "yolo11m-seg.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 4
    patience: int = 20
    project_dir: str = "./runs"
    per_clip_split: bool = True
    train_ratio: float = 0.85


class Settings(BaseModel):
    server: ServerCfg = Field(default_factory=ServerCfg)
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    sam3: Sam3Cfg = Field(default_factory=Sam3Cfg)
    propagate: PropagateCfg = Field(default_factory=PropagateCfg)
    label: LabelCfg = Field(default_factory=LabelCfg)
    augment: AugmentCfg = Field(default_factory=AugmentCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)

    config_path: str | None = None
    repo_root: str = "."

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Settings":
        if path is None:
            cwd_default = Path.cwd() / "configs" / "default.yaml"
            if cwd_default.exists():
                path = cwd_default
        if path is None:
            settings = cls()
        else:
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
            settings = cls(**raw)
            settings.config_path = str(Path(path).resolve())
        settings.repo_root = str(Path.cwd().resolve())
        return settings

    def resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (Path(self.repo_root) / path).resolve()


def load_ontology(path: str | os.PathLike[str]) -> dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"Ontology at {path} must be a flat str→str map")
    return data


def ontology_hash(ontology: dict[str, str]) -> str:
    encoded = json.dumps(ontology, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
