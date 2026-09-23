"""The AVE term: the mean of Euclidean norms, masked.

Two functional forms are easy to get subtly wrong, and this is the first of them
(PipelinePlan.md §4.4):

    AVE = mean over windows of  || v_pred - v_gt ||_2

It is **not** MSE and it is **not** sqrt(MSE). Using MSE silently re-weights the
batch toward the largest errors, so the gradient stops matching the metric's.

Two details that matter more than they look:

* **The norm's derivative is undefined at zero**, so the square root is guarded
  by an epsilon inside the radicand. The bias this introduces is O(eps/|d|) and
  invisible at eps = 1e-12, but the NaN it prevents is not.
* **The denominator is the number of *real* windows.** 72.6% of trajectories are
  shorter than 64 windows, so a batch is mostly padding on drone. Dividing by
  `B*K` instead of `mask.sum()` would quietly scale drone's loss toward zero.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def masked_ave(v_pred: torch.Tensor, v_gt: torch.Tensor,
               mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Mean over the real windows of ||v_pred - v_gt||. Returns 0.0 if none are real."""
    d = v_pred - v_gt
    norm = torch.sqrt((d * d).sum(-1) + eps)          # (B, K)
    m = mask.to(norm.dtype)
    total = (norm * m).sum()
    n = m.sum()
    return total / n.clamp(min=1.0)


def per_window_ave(v_pred: torch.Tensor, v_gt: torch.Tensor) -> torch.Tensor:
    """Unreduced `(B, K)` per-window error, for diagnostics and per-platform tables."""
    return torch.linalg.vector_norm(v_pred - v_gt, dim=-1)
