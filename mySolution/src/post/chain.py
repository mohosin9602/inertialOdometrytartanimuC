"""Post-processing: un-overlap the chunks, then an ordered op list.

**You never build a trajectory here. Integration happens inside the scorer.**
The only places a path is integrated are the training loss and diagnostic plots.

Stitching is the whole of the default chain. Chunks are cut with stride
`chunk_len / 2`, so most windows are predicted twice; each window's predictions
are averaged across the chunks that contain it. Uniform weighting is the
default; a triangular taper that downweights one-sided-context edge windows is a
config option and an ablation (PipelinePlan.md §2.6 step 3).

Two further ops are registered but **off by default, and measured to be worth
almost nothing** (PipelinePlan.md §0.2). They are kept so the report can quote a
number rather than an assumption:

* `zero_clamp` -- a *perfect* zero-velocity oracle is worth 0.0058 score points
  at a 0.05 m/s threshold, 70x smaller than platform routing, and turns actively
  negative above 0.10 m/s. Stationary windows are not rare (36% of human, 34% of
  dog); they are simply worth almost nothing, because a window whose true speed
  is 0.02 m/s can contribute at most 0.02 m/s of error.
* `gain` -- a scalar rescaling of the predictions is worth 0.00024 points at an
  interior optimum of g = 1.025, and every g < 1.0 strictly worsens the score.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- stitching

class PredictionAccumulator:
    """Weighted average of every prediction made for each window.

    `window_ids` is the split's full, ascending id list from the Tier 0 cache,
    which lets a lookup be a `searchsorted` instead of a dict of 30,644 entries.
    """

    def __init__(self, window_ids: np.ndarray, weighting: str = "uniform"):
        self.ids = np.asarray(window_ids, np.int64)
        if not np.all(np.diff(self.ids) > 0):
            raise ValueError("window_ids must be strictly ascending")
        if weighting not in ("uniform", "triangular"):
            raise KeyError(f"unknown stitch weighting {weighting!r}")
        self.weighting = weighting
        self._sum = np.zeros((len(self.ids), 3), np.float64)
        self._wsum = np.zeros(len(self.ids), np.float64)

    def _weights(self, n: int) -> np.ndarray:
        if self.weighting == "uniform":
            return np.ones(n)
        # Bartlett with the endpoints trimmed, so an edge window still carries
        # positive weight and can never end up with a zero denominator.
        return np.bartlett(n + 2)[1:-1] if n > 0 else np.zeros(0)

    def add(self, window_id: np.ndarray, velocity: np.ndarray,
            mask: np.ndarray) -> None:
        """Add one batch: `(B,K)` ids, `(B,K,3)` predictions, `(B,K)` validity."""
        window_id = np.asarray(window_id, np.int64)
        velocity = np.asarray(velocity, np.float64)
        mask = np.asarray(mask, bool)
        for b in range(window_id.shape[0]):
            keep = mask[b]
            ids = window_id[b][keep]
            if ids.size == 0:
                continue
            w = self._weights(int(keep.sum()))
            rows = np.searchsorted(self.ids, ids)
            if not np.array_equal(self.ids[rows], ids):
                missing = ids[self.ids[rows] != ids]
                raise KeyError(f"prediction for unknown window_id(s) {missing[:5]}")
            np.add.at(self._sum, rows, velocity[b][keep] * w[:, None])
            np.add.at(self._wsum, rows, w)

    @property
    def covered(self) -> np.ndarray:
        return self._wsum > 0

    def table(self) -> pd.DataFrame:
        """`window_id, vx, vy, vz`, one row per window, in ascending id order."""
        n_missing = int((~self.covered).sum())
        if n_missing:
            raise RuntimeError(
                f"{n_missing} windows received no prediction. Chunk enumeration "
                f"must cover [0, n) for every trajectory.")
        v = self._sum / self._wsum[:, None]
        return pd.DataFrame({"window_id": self.ids,
                             "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2]})


# --------------------------------------------------------------------- the op list

def _op_stitch(table: pd.DataFrame, **_) -> pd.DataFrame:
    """No-op: stitching already happened in the accumulator, before the chain."""
    return table


def _op_gain(table: pd.DataFrame, g: float = 1.0, **_) -> pd.DataFrame:
    """Scale all predictions by `g`. Measured worth: 0.00024 points at g = 1.025."""
    out = table.copy()
    out[["vx", "vy", "vz"]] *= float(g)
    return out


def _op_zero_clamp(table: pd.DataFrame, threshold: float = 0.05, **_) -> pd.DataFrame:
    """Zero any prediction slower than `threshold`. Measured worth: <= 0.0058 points."""
    out = table.copy()
    v = out[["vx", "vy", "vz"]].to_numpy()
    out.loc[np.linalg.norm(v, axis=1) < float(threshold), ["vx", "vy", "vz"]] = 0.0
    return out


OPS: dict[str, Callable[..., pd.DataFrame]] = {
    "stitch": _op_stitch,
    "gain": _op_gain,
    "zero_clamp": _op_zero_clamp,
}


def run_chain(table: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    """Apply `cfg["post"]["ops"]` in order, each with its own `cfg["post"][op]`."""
    out = table
    for name in cfg["post"]["ops"]:
        if name not in OPS:
            raise KeyError(f"unknown post op {name!r}; have {sorted(OPS)}")
        out = OPS[name](out, **dict(cfg["post"].get(name, {})))
    return out
