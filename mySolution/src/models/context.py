"""Stage 2: let each window see the windows around it. models.md §4.

A single second of inertial signal is genuinely ambiguous. The same accelerometer
reading can mean "accelerating gently on the flat" or "parked on a slope", and
no amount of model capacity resolves that from one second alone. Thirty seconds
of surrounding motion usually does resolve it -- so this stage passes information
along the chunk, giving every window's summary access to its neighbours' before
the velocity is decoded.

**We are allowed to look forward as well as back.** The organizers confirmed that
inference may be non-causal over a whole test trajectory, and nothing in the
rules asks for a model that could run live. The released baseline does have
context -- a one-directional recurrent layer over ten consecutive windows, reset
to zero every ten -- so our advantage is three specific things rather than
"context" in the abstract: **future context, longer context, and no reset.**

**Why M6 makes this the important stage.** Two encoder arms have now been
measured and both say the same thing: removing the whole convolutional encoder
costs car 57%, dog 103% and human 71% of their error but drone only 6.5%, and
adding five times more encoder helps those three and leaves drone alone. Drone's
remaining error is not information sitting unextracted inside the one second.
That leaves the space *between* windows, which is this file.

**The mask is not optional here, and this is where it bites hardest.** Every
stage before this one treats windows independently, so a padded window can only
corrupt itself. This stage mixes windows together, so one padded window can
corrupt every real window in its chunk. 72.6% of trajectories are shorter than 64
windows -- including every single drone flight -- so an unmasked pass would do its
worst damage to the platform the whole milestone is trying to fix.

**One standing caution.** TLIO's own ablation found more context lowered
per-window error without lowering trajectory error, because the leftover errors
became more *correlated* in time -- and correlated error is exactly what
integrating a path punishes. Our data says the same from the other side:
per-trajectory bias is far larger than platform-wide bias. The defence is the
metric-matched trajectory loss at M10, which penalises precisely that
correlation. That makes a testable prediction: context should help the trajectory
half of the score **only when the trajectory loss term is switched on**. M8
crossed with M10 is the one interaction worth measuring on purpose, so read AVE
and ATE20 separately here rather than only the combined score.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .masked import zero_padding

CONTEXT_MODES = ("identity", "bigru", "transformer", "tcn")


DIRECTIONS = ("bidirectional", "causal")


def _bidirectional(direction: str) -> bool:
    """Read the `direction` config key. `causal` is a real ablation, not a leftover.

    Nothing in the competition rules asks for a causal model: the organizers
    confirmed that inference may look forward as well as back over a whole test
    trajectory, so `bidirectional` is the default and the arm we expect to win.
    But a model that only ever looks backwards is the one a robot could actually
    run live, and the technical report is meant to discuss that variant **backed
    by a real run with a real number rather than speculation** (`CLAUDE.md`).
    Making this a switch on both context stages is what turns that requirement
    into a one-line experiment instead of a rewrite.
    """
    if direction not in DIRECTIONS:
        raise KeyError(f"unknown context direction {direction!r}; "
                       f"have {list(DIRECTIONS)}")
    return direction == "bidirectional"


class IdentityContext(nn.Module):
    """Pass window summaries through untouched. The ablation floor.

    It still applies the mask, so that a padded window's summary is exactly zero
    on the way out. Nothing downstream should depend on that -- every reduction
    is masked anyway -- but it means a bug that ignores the mask later shows up
    as an obviously wrong zero rather than as plausible noise.

    **This is the control arm of M8, and it has to be run at the same chunk
    length as the others.** Raising `chunk_len` changes more than the context: it
    changes how batches are composed, how much padding there is, and how many
    optimiser steps make an epoch. Comparing `bigru` at chunk_len 64 against
    `identity` at chunk_len 1 would measure all of that at once and credit it to
    context.
    """

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return zero_padding(z, mask)


class BiGRUContext(nn.Module):
    """A bidirectional recurrent pass along the chunk, added onto the input.

    A recurrent layer walks the chunk one window at a time, carrying a running
    summary as it goes. Running it in both directions and joining the two halves
    means each window's output has been informed by everything before it *and*
    everything after it -- which is allowed here, and is one of the three
    concrete advantages we have over the released baseline.

    **Padding is handled by packing, not by masking afterwards.** PyTorch's
    `pack_padded_sequence` tells the recurrent layer exactly how long each chunk
    really is, so it simply never steps onto a padded window. Masking the output
    instead would be too late: the padded windows would already have polluted the
    running summary that every later real window reads. That distinction is the
    whole reason this class is longer than three lines.

    **The output is added to the input rather than replacing it.** Without that,
    the recurrent layer would have to reproduce everything the encoder already
    worked out before it could add anything of its own, and the M8 comparison
    would partly be measuring how well a GRU can relearn a convolution. With it,
    the stage starts from "change nothing" and learns only what context adds --
    which is the quantity M8 is actually asking about.
    """

    def __init__(self, d_model: int = 256, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1, direction: str = "bidirectional"):
        super().__init__()
        bidirectional = _bidirectional(direction)
        self.direction = direction
        self.gru = nn.GRU(d_model, hidden, num_layers=layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=(dropout if layers > 1 else 0.0))
        # One or two directions of `hidden` come back, and they have to land on
        # `d_model` again for the residual to make sense.
        self.project = nn.Linear((2 if bidirectional else 1) * hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=z.shape[1])
        h = z + self.dropout(self.project(out))
        return zero_padding(self.norm(h), mask)


class SinusoidalPositions(nn.Module):
    """Fixed position signals added to each window's summary. No parameters.

    Attention has no inherent sense of order -- to it a chunk is a bag of windows
    -- so the position of each window has to be supplied as part of its input.
    These are the standard fixed sine and cosine waves of different wavelengths,
    which encode a position as a pattern rather than a number.

    Fixed rather than learned on purpose: `chunk_len` is a swept parameter in this
    milestone, and a learned table would have to be sized for the longest chunk
    and would be undertrained at every other length. Sine waves are defined at
    every position and cost nothing.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        scale = torch.exp(torch.arange(0, d_model, 2).float()
                          * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(pos * scale)
        table[:, 1::2] = torch.cos(pos * scale)
        # A buffer, not a parameter: saved with the weights, never trained.
        self.register_buffer("table", table, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        k = z.shape[1]
        if k > self.table.shape[0]:
            raise ValueError(f"chunk of {k} windows exceeds the positional table "
                             f"({self.table.shape[0]}); raise max_len")
        return z + self.table[:k].unsqueeze(0)


class TransformerContext(nn.Module):
    """Attention across the chunk: every window looks directly at every other.

    Where the recurrent version passes information along a chain, attention lets
    any window read any other in one step, however far apart they are. On a long
    chunk that is a real difference -- the far end of a 256-window chunk is 256
    hops away for a GRU and one hop here.

    **`src_key_padding_mask` is mandatory, not an optimisation.** It tells
    attention which windows do not exist, so no real window ever reads a padded
    one. Leaving it out does not crash and does not look wrong; it just lets
    padding vote on every prediction in the chunk, worst of all on drone. Note
    the sense is inverted from ours: PyTorch wants True where a position should
    be *ignored*, so it is `~mask`.

    LayerNorm-first (`norm_first=True`) is used because it trains stably without
    a carefully tuned warmup, and it keeps this stage consistent with the
    LayerNorm-everywhere rule the rest of the network follows.
    """

    def __init__(self, d_model: int = 256, layers: int = 4, heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, max_len: int = 4096,
                 direction: str = "bidirectional"):
        super().__init__()
        self.causal = not _bidirectional(direction)
        self.direction = direction
        self.positions = SinusoidalPositions(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        # `enable_nested_tensor` is off explicitly: it is a speed path that does
        # not apply with norm_first, and leaving it on only prints a warning.
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.positions(z))
        # A chunk with no real windows at all would make attention average over
        # nothing and return NaN. The dataset never builds one, but a NaN here
        # would spread through the whole batch on the next backward pass, so the
        # guard is worth the one line it costs.
        keep = mask if mask.any(dim=1).all() else mask.clone()
        keep[~mask.any(dim=1)] = True
        # Bool, to match `src_key_padding_mask`: PyTorch deprecates mixing a
        # float attention mask with a bool padding mask. True means "ignore",
        # so the upper triangle above the diagonal is what a causal pass hides.
        attn = (torch.ones(z.shape[1], z.shape[1], dtype=torch.bool,
                           device=z.device).triu(diagonal=1)
                if self.causal else None)
        h = self.encoder(h, mask=attn, src_key_padding_mask=~keep)
        return zero_padding(h, mask)


def build_context(mode: str = "identity", d_model: int = 256,
                  dropout: float = 0.1, **extra) -> nn.Module:
    """Make the context stage named by `mode`. models.md §4 lists the candidates."""
    if mode == "identity":
        # `direction` is meaningless when nothing is mixed, but the reference
        # config carries the key, so accept and discard it rather than making the
        # ablation floor the one mode that refuses the standard config block.
        extra.pop("direction", None)
        return IdentityContext()
    if mode == "bigru":
        return BiGRUContext(d_model=d_model, dropout=dropout, **extra)
    if mode == "transformer":
        return TransformerContext(d_model=d_model, dropout=dropout, **extra)
    if mode == "tcn":
        raise NotImplementedError(
            "context mode 'tcn' is the fourth M8 candidate and is deliberately "
            "not built: models.md §4 recommends baking off 'bigru' against "
            "'transformer' first, since the benchmark says the choice costs "
            "almost nothing either way. Build it only if those two disagree in a "
            "way a dilated, non-causal stack would settle -- and centre-pad it, "
            "as the encoder's TCN does, so it looks forward as well as back.")
    raise KeyError(f"unknown context mode {mode!r}; have {list(CONTEXT_MODES)}")
