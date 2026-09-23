"""Stage 4: use the embodiment vector to adjust how velocity is decoded. models.md §6.

The embodiment branch works out *what kind of machine this is*. The conditioner
is what that opinion actually does to the prediction. All the options share one
interface -- `(features, e) -> features` -- which is what makes swapping them a
single config line rather than a rewrite.

| mode      | what it does                                          | when      |
| --------- | ----------------------------------------------------- | --------- |
| `concat`  | glue `e` onto every window's summary as extra numbers  | M5 start  |
| `film`    | let `e` set a per-channel scale and offset             | M9 default|
| `moe`     | several decoders, blended by a soft gate               | late M9   |

**Why `concat` first and `film` later.** Concatenation is the simplest thing
that could possibly work, so it is the right floor for the M9 comparison to
measure against. But expect it to be *weak* here, and for a specific reason: the
embodiment vector arrives as a handful of extra numbers alongside a couple of
hundred others, gets multiplied by one weight matrix, and is then diluted. That
gives the branch that produced it very little gradient to learn from. FiLM
reaches *inside* the decoder and scales every channel, so the same vector has a
much larger effect and receives a much stronger training signal.

**Why FiLM starts as a do-nothing operation.** The last layer of the small
network that produces the scale and offset is initialised to zero, so training
begins with scale = 1 and offset = 0 -- exactly as if there were no conditioning
at all. The model therefore learns to predict velocity first and picks up
conditioning as a correction on top. That is more stable, and it makes the
ablation cleaner, because the conditioned and unconditioned runs start from
literally the same place.

**One failure mode, and it belongs to `moe` alone.** With several expert
decoders and a soft gate, a gate trained on unbalanced data can send everything
to one expert and still lower the loss -- drone is only 17% of training windows,
so ignoring drone is cheap. The loss curve looks perfectly healthy while this
happens. Two defences when `moe` is built: always pair it with the
platform-balanced batch sampler, and log how much traffic each expert receives
every epoch. `film` has no experts to collapse, which is exactly why it is the
default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CONDITIONER_MODES = ("none", "concat", "film", "moe")


class NoConditioning(nn.Module):
    """Ignore the embodiment vector entirely. The ablation floor for M9."""

    def forward(self, features: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return features


class ConcatConditioner(nn.Module):
    """Append `e` to every window's summary, then mix back down to `d_model`.

    `e` describes the whole chunk, so the same vector is copied onto each of the
    chunk's K windows before the concatenation. The linear layer afterwards is
    what lets the network decide how much of the summary and how much of the
    embodiment to keep.
    """

    def __init__(self, d_model: int = 256, width: int = 32):
        super().__init__()
        self.mix = nn.Linear(d_model + width, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        b, k, _ = features.shape
        broadcast = e.unsqueeze(1).expand(b, k, e.shape[-1])
        return self.norm(self.mix(torch.cat([features, broadcast], dim=-1)))


class FiLMConditioner(nn.Module):
    """Let `e` choose a multiplier and an offset for each feature channel.

    "FiLM" is feature-wise linear modulation: a small side network turns `e` into
    two vectors, one that scales the features and one that shifts them. Because
    the input is continuous, an uncertain embodiment produces a setting part-way
    between two platforms rather than a hard choice between them.

    The zero-initialised output layer is what makes this start as the identity;
    see the module docstring for why that matters.
    """

    def __init__(self, d_model: int = 256, width: int = 32, hidden: int = 64):
        super().__init__()
        self.side = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        nn.init.zeros_(self.side[-1].weight)
        nn.init.zeros_(self.side[-1].bias)
        self.d_model = d_model

    def forward(self, features: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.side(e)                                  # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)                  # (B, D) each
        # 1 + gamma so that a zero output means "leave the features alone".
        return features * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


def build_conditioner(mode: str = "concat", d_model: int = 256,
                      width: int = 32, **extra) -> nn.Module:
    """Make the conditioner named by `mode`. models.md §6 lists the ladder.

    **Why `insert_at` is dropped here rather than deleted from `config.py`.**
    Every config inherits a dead `conditioner.insert_at: []` from `_M0_ZERO`.
    `ConcatConditioner` takes no `**extra`, so it has silently ignored the key
    since M0; `FiLMConditioner` does, so forwarding it raised a `TypeError` and
    made M9's headline arm unlaunchable from the promoted reference. Removing
    the key at source would be tidier and is the wrong fix: the config hash
    covers every key, so it would rehash all twenty runs in `results.jsonl`,
    breaking duplicate detection and the report's ablation tables at once.
    """
    extra.pop("insert_at", None)
    if mode == "none":
        return NoConditioning()
    if mode == "concat":
        return ConcatConditioner(d_model, width)
    if mode == "film":
        return FiLMConditioner(d_model, width, **extra)
    if mode == "moe":
        raise NotImplementedError(
            "a soft mixture of experts is the last rung of the M9 ladder "
            "(models.md §6). It must not be built without the platform-balanced "
            "sampler and a per-expert gate-mass log, because expert collapse "
            "looks exactly like a healthy loss curve.")
    raise KeyError(f"unknown conditioner mode {mode!r}; have {list(CONDITIONER_MODES)}")
