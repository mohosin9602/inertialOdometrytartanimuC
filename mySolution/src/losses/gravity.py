"""The auxiliary gravity term: teach the trunk which way is down. M15.

The gravity head predicts, for every window, a unit vector pointing the way
gravity does in the body's own coordinates. This term scores that prediction
against the truth, and the resulting gradient is the point: it asks the trunk
to carry tilt information that the velocity loss alone never made worth the
capacity on human and car (CLAUDE.md finding 15).

**The target is computed, not cached.** The batch already carries `q_gt`, the
mid-window ground-truth quaternion, and the down-direction is the third row of
its rotation matrix (`prep.orientation.gravity_from_quaternion_torch`). Mid-window
sampling is the same convention CLAUDE.md finding 3 pins for the scorer.

**Why this is legal.** Ground-truth `quat` ships with train and val, and the
organizers confirmed it may be used as an auxiliary training target. Only the
*test* labels are withheld, and test rows carry no quaternion at all.

**Which rows count.** `q_gt` is all *zeros* -- not a sentinel -- on test rows
and on padding, so there is no `-1` to filter on the way the platform term has.
The quaternion's own length is the filter: a real one has length 1, a missing
one length 0, and `|q|^2 > 0.5` separates them with room to spare. It is
combined with `mask` so a padded window can never count even if something ever
wrote into its slot. A batch with no valid rows contributes exactly 0.0 and no
NaN -- the platform term's contract, for the same reason.

**The loss is `1 - cos`, not the angle.** For unit vectors it equals half the
squared distance between them, it is smooth everywhere, and it has a useful
gradient even at 180 degrees. `arccos` is non-smooth at both ends and buys
nothing. The angle is still reported, as a diagnostic in degrees, because it is
what the kill criterion reads: below 10 degrees macro-median or the head has not
learned the quantity at all (the filter itself manages 2.86).
"""
from __future__ import annotations

import math

import torch

from ..prep.orientation import gravity_from_quaternion_torch


def gravity_valid(q_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`(B, K)` bool: windows that are real AND carry a ground-truth quaternion.

    Checked on the raw quaternion's squared length, before anything divides by
    it -- normalising first would divide by zero on exactly the rows this
    exists to exclude.
    """
    return (q_gt.pow(2).sum(-1) > 0.5) & mask.bool()


def _cosines(g_pred: torch.Tensor, q_gt: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """Cosine between prediction and truth on the valid windows, as a flat vector."""
    valid = gravity_valid(q_gt, mask)
    target = gravity_from_quaternion_torch(q_gt.to(g_pred.dtype))
    return (g_pred * target).sum(-1)[valid]


def gravity_cosine_loss(g_pred: torch.Tensor, q_gt: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Mean of `1 - cos(g_pred, g_true)` over valid windows; exactly 0.0 if none."""
    cos = _cosines(g_pred, q_gt, mask)
    if cos.numel() == 0:
        return g_pred.new_zeros(())
    return (1.0 - cos).mean()


def gravity_angles_deg(g_pred: torch.Tensor, q_gt: torch.Tensor,
                       mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-window angle to truth in degrees, and the `(B, K)` mask it was taken over.

    The flat angle vector lines up with `valid.nonzero()`, so a caller that
    needs to know which platform each angle came from can index the batch's
    `platform_id` with the same mask.
    """
    valid = gravity_valid(q_gt, mask)
    with torch.no_grad():
        cos = _cosines(g_pred.detach(), q_gt, mask).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(cos)), valid


def gravity_angle_deg(g_pred: torch.Tensor, q_gt: torch.Tensor,
                      mask: torch.Tensor) -> float:
    """Mean angle to truth over valid windows, in degrees; NaN if there are none.

    A per-batch diagnostic for the training log, the way `platform_accuracy` is.
    The number the kill criterion reads is the per-platform *median* on val,
    computed in `pipeline.predict_dataset` from `gravity_angles_deg`.
    """
    angles, _ = gravity_angles_deg(g_pred, q_gt, mask)
    return float(angles.mean()) if angles.numel() else math.nan
