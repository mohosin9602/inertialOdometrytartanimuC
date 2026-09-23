"""The conditioning vector actually fed to the model. PipelinePlan.md §4.1.

RULE CHANGE, 2026-08-30 -- read this before using anything here on a submission
path. The host clarified that predictions must come from a single model with one
shared set of weights, and that recovering the platform in order to route to
per-platform experts is against the rules. That retires the contract this module
was written for: an external classifier producing a cached posterior that the
velocity model consumes as data is exactly the forbidden shape.

The replacement lives inside the network -- `models.Embodiment` pools the
model's own window embeddings into a continuous embodiment vector, and
`models.Conditioner` reads that. Nothing in this module runs on a submission
path any more.

The default source is now `learned`, which puts **nothing** in the batch: the
model builds its own embodiment vector from the signal and ignores the `cond`
slot entirely. `tests/test_m5.py` proves that by scrambling `cond` and checking
that not one prediction moves.

Every other source here is a DIAGNOSTIC, and all three of them are gated behind
`run.diagnostic_only` -- `pooled_posterior` included, since 2026-08-31. It is
the one that would be most tempting to reach for by accident, because it does
not read a label directly; it reads the *output of a model that read labels*,
which is the same thing one step removed and is exactly the shape the
clarification names. See STATUS.md "Rule change, 2026-08-30" and
PipelinePlan.md 4.1.

--- original contract, retained for context ---

At test time the platform posterior is pooled over an entire trajectory and
arrives at the velocity model looking essentially one-hot. The contract is that
**training sees the same distribution inference sees, by construction**: train
the classifier standalone, run it over every train/val trajectory, pool exactly
as inference will, cache that, and condition on the cached vectors -- never on
the true one-hot label and never on a per-window posterior.

`cond.source` values:

    learned            the default. Zeros -- the model builds its own vector
    uniform            0.25 everywhere -- the pre-M5 sentinel, reads no label
    pooled_posterior   cached per-trajectory 4-vector from M2. GATED
    true_label         diagnostic ceiling. GATED: needs run.diagnostic_only
    corrupted_onehot   true label flipped at a chosen rate; an ablation. GATED
    per_window         the broken variant, kept so the report can quantify it

Reading the true platform into `cond` on any evaluation path -- **including
val** -- is precisely the defect that makes the published 0.637 baseline
unreproducible. The gate below is what stops that happening by accident, and the
`diagnostic_only` flag it demands is stamped into every downstream result.
"""
from __future__ import annotations

import numpy as np

from ..paths import PLATFORMS

SOURCES = ("learned", "uniform", "pooled_posterior", "true_label",
           "corrupted_onehot", "per_window")

#: Sources that carry platform identity into the batch from outside the network.
#: Using any of them demands `run.diagnostic_only`, and that flag is stamped
#: into the results record so a ceiling experiment can never be quietly compared
#: against a real run. `pooled_posterior` joined this list on 2026-08-31: it does
#: not read a label directly, it reads the output of a classifier that did, and
#: the 2026-08-30 clarification forbids exactly that shape on a submission path.
GATED = ("true_label", "corrupted_onehot", "pooled_posterior")
N_PLATFORMS = len(PLATFORMS)


class CondError(RuntimeError):
    """Raised when a conditioning source would leak a label into inference."""


def _onehot(platform_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(platform_ids, dtype=np.int64)
    if (ids < 0).any():
        raise CondError("a one-hot cond was requested but some platform_id is -1 "
                        "(the test split is anonymised)")
    out = np.zeros((len(ids), N_PLATFORMS), np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def build_cond(source: str, platform_ids: np.ndarray, *,
               diagnostic_only: bool = False,
               posteriors: dict[str, np.ndarray] | None = None,
               traj_ids: list[str] | None = None,
               corruption_rate: float = 0.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the `(B, 4)` float32 conditioning vector for a batch."""
    if source not in SOURCES:
        raise KeyError(f"unknown cond.source {source!r}; have {list(SOURCES)}")
    b = len(platform_ids)

    if source in GATED and not diagnostic_only:
        raise CondError(
            f"cond.source={source!r} reads the true platform label. It is allowed "
            f"only when run.diagnostic_only is True, and the flag is then stamped "
            f"into the result so the run can never be quietly compared against a "
            f"real one. See PipelinePlan.md §2.5 and §4.1.")

    if source == "learned":
        # Deliberately zeros, and deliberately still shape (B, 4): the batch
        # contract is fixed, and a model that builds its own embodiment vector
        # ignores this slot. Zeros rather than 0.25 so that a model which
        # accidentally *did* read it would produce visibly broken output rather
        # than something plausible.
        return np.zeros((b, N_PLATFORMS), np.float32)
    if source == "uniform":
        return np.full((b, N_PLATFORMS), 1.0 / N_PLATFORMS, np.float32)
    if source == "true_label":
        return _onehot(platform_ids)
    if source == "corrupted_onehot":
        rng = rng or np.random.default_rng(0)
        ids = np.asarray(platform_ids).copy()
        flip = rng.random(b) < corruption_rate
        ids[flip] = rng.integers(0, N_PLATFORMS, int(flip.sum()))
        return _onehot(ids)
    if source == "pooled_posterior":
        if posteriors is None or traj_ids is None:
            raise CondError(
                "cond.source='pooled_posterior' needs the cached per-trajectory "
                "posteriors from prep.derived.posteriors, which land at M2.")
        return np.stack([posteriors[t] for t in traj_ids]).astype(np.float32)

    raise NotImplementedError(
        "cond.source='per_window' is retained for the report's ablation of the "
        "broken variant and is not built yet (PipelinePlan.md §4.1).")
