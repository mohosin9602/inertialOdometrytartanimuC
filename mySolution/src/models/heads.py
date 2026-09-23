"""Stage 5: the output layers. models.md §6.

Three heads exist today. The first two have very different jobs; the third
(`GravityHead`, M15) is an auxiliary signal like the second, and is documented
on its class.

**`velocity` is the only thing that is ever submitted.** Three numbers per
window: the body-frame velocity, in metres per second. Both halves of the
competition score are pure functions of this one output.

**`platform` is never read at inference.** It predicts which of the four
platforms the chunk came from, and it exists purely so that gradient flows back
into the embodiment vector and organises it. Training on the platform labels is
legal because those labels ship with the train and val data -- supervising on
published labels is modelling, not leakage; only the *test* labels are withheld.
The compliance rule is mechanical and testable: **deleting this head at
inference must not change a single prediction.** `tests/test_m5.py` asserts
exactly that.

**There is deliberately no attitude head, and there never will be.** The scorer
rotates our predicted body-frame velocities into the world using *its own*
ground-truth attitude, which it holds and we never submit. An orientation output
would be pure wasted capacity. `GravityHead` is not an exception: it predicts
tilt only so its gradient can shape the trunk, and its output goes nowhere.

**`direction_speed` was sketched here as an M9 ablation and is now deliberately
NOT built.** M9 arm 0 measured it away on 2026-09-06; the reasoning is worth
keeping, because the conclusion generalises past this competition.

Its motivation was that on car, dog and drone the baseline's predicted direction
is essentially random -- around 90 degrees from truth, which is what guessing
gives -- while its speed is merely wrong by a factor. Splitting the output into
"which way" and "how fast" was meant to stop directional uncertainty leaking out
as magnitude collapse, which is what a norm-based loss makes a confused model do.
This docstring already guessed the outcome: "the root cause is platform identity,
which this design fixes properly, so the split may buy nothing once conditioning
works."

That guess was right, but not for the expected reason, and the real reason is the
interesting one. The coupling it targets **is still there** -- drone's speed ratio
still falls from 0.97 on well-directed windows to 0.71 on windows more than 90
degrees off. What settles it is the *optimal* gain inside each angular bucket:
**0.000 beyond 90 degrees**, on every platform and every seed. The correct
response to a badly-directed window is to predict nothing at all, and the model
already shrinks too little rather than too much. A head that decouples speed from
directional confidence would push the wrong way.

This is a property of the metric, not of this model. Under a mean-of-norms loss a
full-magnitude guess in a uniformly random 3-D direction costs about 1.33 |v|
while predicting zero costs exactly |v|, so shrinking is the optimal hedge.
**Any architecture that keeps the speed branch confident while the direction
branch is unsure is fighting AVE.** CLAUDE.md finding 14; reproduce with
`python -m src.evaluation.decompose <run> --oracles`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VelocityHead(nn.Module):
    """Per-window body-frame velocity, `(B, K, D) -> (B, K, 3)`.

    The LayerNorm before the final layer keeps the input to that layer at a
    steady scale no matter what the conditioner did to the features, which
    matters because FiLM can legitimately multiply them by a large factor.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(features))


class PlatformHead(nn.Module):
    """Four platform scores from the embodiment vector, `(B, E) -> (B, 4)`.

    Auxiliary training signal only. Nothing on the velocity path reads its
    output -- see this module's docstring for why that is a rule rather than a
    preference.

    It is also a free diagnostic. Logging its accuracy every epoch says directly
    whether the network is learning to tell the platforms apart. If that
    accuracy is not climbing toward the ~99% a plain gradient-boosting model
    reaches on hand-built features, then the conditioning cannot be working, and
    the log says which half of the design to go and look at.
    """

    def __init__(self, width: int = 32, n_platforms: int = 4):
        super().__init__()
        self.out = nn.Linear(width, n_platforms)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.out(e)


class GravityHead(nn.Module):
    """Per-window down-direction in body coordinates, `(B, K, D) -> (B, K, 3)`, unit length.

    M15, the auxiliary gravity head. **Auxiliary training signal only**, exactly
    like `PlatformHead`: nothing on the velocity path reads its output, so
    deleting it changes no prediction (`tests/test_m15.py`). Its whole job is the
    gradient it sends back into the trunk.

    **Why it exists.** Finding 15 measured the three explicit gravity channels at
    0.0179 of score, two thirds of it on human, and almost nothing on drone --
    because a drone flight fits inside one 64-window chunk and the bidirectional
    context stage works the tilt out from the gyroscope unaided. The head asks
    whether a *gradient* can make the trunk do for human and car what it already
    does for drone, on six raw channels. It is the "can supervision substitute
    for the extra channels?" rung of the ablation ladder in
    `.claude/GravityDirectionPlan.md`, not a score lever: once the channels are an
    input, predicting gravity is trivial and the head would learn a copy.

    **It reads the context output, before the conditioner.** Not the encoder
    output -- one second of signal cannot know its own tilt history, which is the
    whole value of the filter. And not after FiLM, whose per-platform scale and
    offset is a nuisance transformation for a geometric quantity.

    1,283 parameters at `d_model = 256`: `LayerNorm` 512 + `Linear(256, 3)` 771.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Only the direction carries meaning, so the output is put on the unit
        # sphere by construction rather than asked to learn a length of one.
        return nn.functional.normalize(self.out(self.norm(features)), dim=-1)
