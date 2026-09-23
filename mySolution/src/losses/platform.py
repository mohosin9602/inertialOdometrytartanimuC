"""The auxiliary platform term: teach the embodiment vector what it is for.

The embodiment branch pools the chunk's window summaries into one continuous
vector. Nothing forces that vector to be *about* the platform -- left to itself,
it might encode speed, or noise level, or nothing much at all. This term is what
organises it: a small classifier reads the vector, tries to name the platform,
and the resulting gradient pushes the vector into a shape where the four
platforms sit apart from each other.

**Why this is legal.** It trains on the platform labels that ship with the train
and val data. Supervising on published labels is modelling, not leakage -- only
the *test* labels are withheld, and nothing here touches those. The 2026-08-30
clarification forbids recovering the platform *outside* the network in order to
route between per-platform experts; it explicitly encourages a single network
that adapts internally, which is exactly what this trains.

**Two failure directions to watch when tuning its weight.** Too small and the
pooled vector never organises itself by embodiment, so the conditioner has
nothing useful to condition on. Too large and the network spends its capacity on
a classification problem that a decision tree already solves perfectly, at the
expense of the velocity accuracy that is actually scored. The default is 0.1 and
M9 sweeps it.

**The test split has no labels**, so `platform_id` is -1 there. Those rows are
dropped rather than treated as a fifth class; a batch with no labelled rows
contributes exactly zero and no NaN.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def platform_cross_entropy(logits: torch.Tensor,
                           platform_id: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of `(B, 4)` scores against `(B,)` labels, ignoring -1 rows.

    Returns exactly 0.0 when nothing in the batch is labelled, so an unlabelled
    batch neither crashes nor quietly nudges the weights.
    """
    labelled = platform_id >= 0
    if not bool(labelled.any()):
        return logits.new_zeros(())
    return F.cross_entropy(logits[labelled], platform_id[labelled])


def platform_accuracy(logits: torch.Tensor, platform_id: torch.Tensor) -> float:
    """Share of labelled rows whose highest-scoring platform is the right one.

    A diagnostic, not a loss. Logging it every epoch says directly whether the
    network is learning to tell the platforms apart, which is the precondition
    for the conditioning to be doing anything at all.
    """
    labelled = platform_id >= 0
    if not bool(labelled.any()):
        return float("nan")
    predicted = logits[labelled].argmax(dim=-1)
    return float((predicted == platform_id[labelled]).float().mean())
