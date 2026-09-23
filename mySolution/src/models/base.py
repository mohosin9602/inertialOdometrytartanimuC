"""The velocity model interface every model in this project implements.

    forward(x, mask, cond) -> {"velocity": (B, K, 3), ...}

**The velocity path may read `x`, `mask` and `cond`. Nothing else.**
`platform_id`, `q_gt`, `p_gt` and `v_gt` are visible to losses and metrics and
never to the forward pass. That is not a style preference: reading the true
platform into an inference path is exactly the defect that makes the published
0.637 baseline unreproducible by any participant. `forward_batch` below is the
enforcement -- it is the only way the runner calls a model, and it hands over a
three-key subset of the batch, so a model physically cannot reach a label.

Shapes are fixed by PipelinePlan.md §2.3. Any reshaping a particular
architecture prefers happens inside that architecture, never in the dataloader.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import torch
import torch.nn as nn

MODEL_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


class VelocityModel(nn.Module, ABC):
    """Base class. Subclasses implement `forward(x, mask, cond)`."""

    #: Every model declares how many input channels it expects, so a config
    #: mismatch with prep.features surfaces at construction, not at step 400.
    in_channels: int = 6

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x` (B,K,C,T), `mask` (B,K) bool, `cond` (B,4) -> {"velocity": (B,K,3)}."""

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Mapping, in_channels: int = 6) -> VelocityModel:
    """Instantiate the model named by `cfg["model"]["kind"]`."""
    model_cfg = dict(cfg["model"])
    kind = model_cfg.pop("kind")
    if kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown model kind {kind!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[kind](in_channels=in_channels, **model_cfg)


def forward_batch(model: VelocityModel, batch: Mapping) -> dict[str, torch.Tensor]:
    """Call a model with exactly the three keys it is allowed to see.

    **The velocity comes back in whatever frame the input was in.** When
    `data.input_repr` is one of the `aligned` representations the input has been
    rotated into a gravity-aligned frame, so the model predicts there too, and
    this function rotates the prediction back into the body frame before anybody
    else touches it. Doing it here, in one place, is what keeps the targets, the
    losses, the ATE20 term, the scorer and the submission writer completely
    unaware that alignment exists.

    The rotation is a fixed orthogonal change of variables carried in the batch
    (`align_R`, body -> aligned, built by the complementary filter from the IMU
    alone, so it is legal on test). The gradient flows through it, which is what
    makes training in the aligned frame equivalent to rotating the target: for an
    orthogonal `R`, minimising `||R^T v_hat - v||` IS minimising `||v_hat - R v||`.

    `align_R` is the identity for every non-aligned representation, so this is a
    no-op -- verified bit-exact -- for every run made before it existed. The
    model itself is still handed exactly x, mask and cond.
    """
    out = model(x=batch["x"], mask=batch["mask"], cond=batch["cond"])
    if "velocity" not in out:
        raise KeyError(f"{type(model).__name__}.forward must return a 'velocity' key; "
                       f"got {sorted(out)}")
    b, k = batch["mask"].shape
    if tuple(out["velocity"].shape) != (b, k, 3):
        raise ValueError(f"velocity must be (B,K,3) = {(b, k, 3)}, "
                         f"got {tuple(out['velocity'].shape)}")
    R = batch.get("align_R")
    if R is not None:
        # v_body = R^T v_aligned. The transpose is in the subscripts, not a
        # materialised copy.
        out = {**out, "velocity": torch.einsum(
            "bkji,bkj->bki", R.to(out["velocity"].dtype), out["velocity"])}
    return out
