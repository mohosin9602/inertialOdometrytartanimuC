"""Stage 3: the model works out what kind of machine it is riding on. models.md §5.

**Why this is the most important component in the project.** The four platforms
move in completely different ways, and a model that does not know which one it is
looking at cannot predict any of them well. Measured: the released baseline
scores 0.4185 when it is told the platform and 0.8346 when it is not -- a gap of
0.42 score points, which is larger than every other lever in the project put
together.

**Why it has to happen inside the network.** The competition's rules were
clarified on 2026-08-30: predictions must come from a single model with one
shared set of weights, and running a separate classifier to recover the platform
so that windows can be routed to per-platform experts is explicitly against the
rules. A model that adapts internally -- learned conditioning, a soft mixture of
experts inside one network -- is explicitly encouraged. So the design is: the
network forms its own opinion about the platform, from the same six or nine
channels it gets at test time, and uses that opinion to adjust itself.

**The information really is free; the mechanism is the work.** A plain
gradient-boosting model on 40 hand-built statistics identifies 475 of 475
labelled trajectories correctly, and so do a single decision tree, nearest
neighbours, and logistic regression once a whole trajectory is pooled. The four
platforms are almost linearly separable in a space of simple window statistics,
and a learned convolutional encoder computes strictly richer statistics than
those. So the branch should recover this nearly for free.

**Why a continuous vector rather than four probabilities.** Zero mistakes in 475
does not prove perfection -- statistically, expect zero or one misidentified
trajectory among the 89 test ones. A four-way decision forces a confident,
possibly wrong commitment on exactly those cases. A continuous vector lets an
ambiguous recording settle somewhere between two platforms instead, which
degrades gracefully. Its width is a config knob and its axes need not correspond
to the four platform names at all.

**Pooling inside the chunk is enough; no second pass over the trajectory.**
Identification saturates almost immediately -- 99.36% of trajectories are placed
correctly from a single window and 99.90% from four. Every planned chunk length
is far above four.

One honest caveat about M5 specifically. It runs at `chunk_len = 1`, where there
is a single window and therefore nothing to pool. Expect roughly the
single-window regime there, and expect it to improve as soon as M8 raises the
chunk length. A disappointing conditioning result at M5 is not evidence against
the design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .masked import masked_mean

POOL_MODES = ("mean", "attention")


class Embodiment(nn.Module):
    """Pool the chunk's window summaries into one continuous embodiment vector.

        z (B, K, D) + mask (B, K)  ->  e (B, E)

    The pooling is a masked average, so padded windows contribute nothing. That
    is not a detail: every drone flight is short enough that its chunks are
    mostly padding at the larger chunk lengths, and an unmasked average would
    hand the branch a vector made largely of zeros for the one platform whose
    identity matters most.

    The final LayerNorm keeps `e` at a predictable scale from the very first
    step, which in turn keeps the conditioning parameters it drives in a sane
    range instead of needing to be trained out of a bad start.
    """

    def __init__(self, d_model: int = 256, width: int = 32,
                 hidden: int = 128, pool: str = "mean"):
        super().__init__()
        if pool not in POOL_MODES:
            raise KeyError(f"unknown embodiment pool {pool!r}; have {list(POOL_MODES)}")
        if pool == "attention":
            raise NotImplementedError(
                "attention pooling is an M9 ablation (models.md §5). "
                "'mean' is the default and is what M5 uses.")
        self.pool = pool
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.LayerNorm(width),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_mean(z, mask, dim=1)        # (B, D)
        return self.net(pooled)                     # (B, E)
