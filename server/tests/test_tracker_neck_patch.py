"""Verify tracker_neck weights are aliased correctly from
`tracker_model.tracker_neck.*` in the on-disk safetensors to top-level
`tracker_neck.*` on the live `Sam3VideoModel`.

Skipped if the local checkpoint is absent (CI runs without weights)."""

from __future__ import annotations

from pathlib import Path

import pytest

CKPT_DIR = Path(__file__).resolve().parents[2] / "sam3_checkpoint_hf"
CKPT_FILE = CKPT_DIR / "model.safetensors"


@pytest.mark.skipif(
    not CKPT_FILE.is_file(),
    reason=f"sam3 checkpoint not present at {CKPT_FILE}",
)
def test_patch_tracker_neck_loads_22_weights() -> None:
    import torch  # local import — heavy, skip if not installed
    from safetensors.torch import safe_open
    from transformers import Sam3VideoModel

    from tk_vision.annotate.sam3 import Sam3Engine

    model = Sam3VideoModel.from_pretrained(str(CKPT_DIR), dtype=torch.float32)

    # Pre-patch: tracker_neck weight should NOT match the checkpoint source.
    pre = model.tracker_neck.fpn_layers[0].proj1.weight.detach().clone()
    with safe_open(str(CKPT_FILE), framework="pt", device="cpu") as f:
        src = f.get_tensor("tracker_model.tracker_neck.fpn_layers.0.proj1.weight").float()
    assert (pre - src).abs().max().item() > 1e-2, (
        "tracker_neck already matches checkpoint pre-patch — test setup wrong "
        "or HF auto-loader resolved this in a version we didn't expect."
    )

    n = Sam3Engine._patch_tracker_neck(model)
    assert n == 22, f"expected 22 weights patched, got {n}"

    post = model.tracker_neck.fpn_layers[0].proj1.weight.detach().float()
    assert (post - src).abs().max().item() < 1e-4, (
        "post-patch tracker_neck does not match checkpoint within fp32 tolerance"
    )
