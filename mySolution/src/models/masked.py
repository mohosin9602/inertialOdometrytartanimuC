"""Reductions that respect the padding mask. Tier 1.

**Why this file exists at all.** A batch is a rectangle, but trajectories are
not. 72.6% of train and val trajectories are shorter than 64 windows -- every
one of the 337 drone flights, plus 16% of dog -- so most batches are part real
data and part zeros used as filler. Averaging over the rectangle silently mixes
those zeros in, and it does the most damage to the shortest trajectories, which
is to say **drone: the platform with the fewest windows and the worst baseline
error**. The bug would not crash anything and would not look wrong on a loss
curve. It would just quietly cost the competition's hardest quarter.

So every reduction over the window axis goes through one of these two helpers,
and `tests/test_m5.py` checks that poisoning the padded entries with large
numbers changes no result by a single bit.
"""
from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean over `dim`, counting only the entries `mask` marks as real.

    `x` is `(B, K, D)` and `mask` is `(B, K)` boolean in the usual case.
    Rows with no real entries at all return zeros rather than dividing by zero;
    an all-padding sample must produce a finite number, not a NaN that spreads
    through the whole batch on the next backward pass.
    """
    m = mask.to(x.dtype).unsqueeze(-1)              # (B, K, 1)
    total = (x * m).sum(dim=dim)
    count = m.sum(dim=dim).clamp(min=1.0)
    return total / count


def zero_padding(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Force padded positions of `(B, K, D)` to exactly zero.

    Used just before a reduction that cannot take a mask itself, and after any
    layer that mixes information across the window axis. Multiplying rather than
    assigning keeps the operation differentiable and keeps the gradient at
    padded positions at zero, which is what we want -- padding must never move
    a weight.
    """
    return x * mask.to(x.dtype).unsqueeze(-1)
