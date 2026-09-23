"""The M0 model: a learnable constant, initialised to zero.

This is deliberately *not* a special-cased shortcut in the runner. It is a real
`VelocityModel` with a real parameter and a real gradient path, plugged into the
same interface every later model uses, so pushing it end to end exercises the
plumbing rather than a bypass of it. With the bias at its zero initialisation it
predicts exactly the all-zeros submission, whose val score is a known constant
(1.0144) -- which is what makes M0 falsifiable.

Note the connection to PipelinePlan.md §0.3: a prediction of exactly zero makes
all three singular values of the ATE20 alignment equal, and plain autograd then
produces NaN on every segment. A zero-initialised head hits that on its first
backward pass. The fix is detaching the Umeyama rotation and translation, exact
by the envelope theorem. That trap is why this model, not a random one, is the
right thing to keep around once the metric loss lands at M10.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import VelocityModel, register


@register("constant_zero")
class ConstantZeroModel(VelocityModel):
    """Predicts one learned body-frame velocity for every window.

    Left at its initialisation it is the all-zeros submission; trained, it is
    the "predict the global mean velocity" reference. The organizers' own table
    records that a per-platform constant-velocity submission scores 1.060 --
    *worse* than all-zeros at 1.000 -- because constant velocity integrates into
    a straight line, which ATE20 punishes harder than integrating into a point.
    """

    def __init__(self, in_channels: int = 6, init: float = 0.0, **unused):
        super().__init__()
        self.in_channels = in_channels
        self.bias = nn.Parameter(torch.full((3,), float(init)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cond: torch.Tensor) -> dict[str, torch.Tensor]:
        b, k = mask.shape
        if x.shape[2] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, "
                             f"got {x.shape[2]}")
        # Padded positions are NOT zeroed here. Models do not owe the rest of the
        # pipeline a mask; every downstream reduction is required to apply one,
        # and silently zeroing padding would hide exactly the bug the mask exists
        # to catch.
        return {"velocity": self.bias.view(1, 1, 3).expand(b, k, 3)}
