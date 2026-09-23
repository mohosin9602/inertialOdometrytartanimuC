"""The ATE20 term: the metric itself, differentiable. PipelinePlan.md §0.3, §4.4.

    ATE20 = mean over segments of  RMS( aligned integrated path - ground truth )

**This is not an approximation of the competition metric, it is the metric.**
The scorer rotates predicted *body-frame* velocities into the world using
ground-truth attitude, and train and val ship the very same quaternions, so
every input the scorer uses is available at training time. What is reproduced
here is `kaggle_metric_tartanimu_score._ate_segment`, line for line, in batched
PyTorch.

Four things about it are not obvious, and three of them were measured before a
line of this file existed (CLAUDE.md finding 7).

**1. Detach the alignment.** Umeyama returns the `(R, t)` that *minimises* the
very quantity being differentiated, so by the envelope theorem the derivative of
the loss through `R` and `t` is exactly zero -- dropping it changes no gradient
and is not an approximation. It was verified against finite differences to
1.2e-08 and makes the backward pass 2.8x faster.

**2. Detaching is also the only fix for the NaN.** The danger with a singular
value decomposition is not a *zero* singular value, it is two *equal* ones: the
gradient contains `1/(s_i^2 - s_j^2)`. A prediction of exactly zero makes the
cross-covariance zero, all three singular values equal, and plain autograd
produces NaN on **617 of 617 val segments**. A zero-initialised velocity head
hits this on its first backward pass, so this is the normal case rather than an
edge case. A ridge does not fix it; detaching removes the SVD from the graph
entirely and fixes it completely and for free.

**3. Run it on CPU in float64 even when the model trains on MPS.** 0.80 ms
against 6.56 ms for a 16x64 batch, and the bottleneck is not the SVD -- it is
`torch.linalg.det`, which costs 63.6 ms on a (112,3,3) batch on MPS against
0.186 ms on CPU. The closed-form 3x3 cofactor expansion below sidesteps it. The
predictions cross the device boundary mid-graph, which is a differentiable copy
in PyTorch, but a silently zeroed gradient there would look exactly like "the
ATE20 term does not help", so `tests/test_m10.py` gradchecks the boundary rather
than trusting it.

**4. A boundary window belongs to two segments.** Segments tile the travelled
distance without overlap, but the scorer sets `start = end` and slices `[s:e+1]`
inclusive, so the shared window contributes to both segment errors and
accumulates gradient twice. That is correct behaviour to reproduce, which is why
this file takes explicit inclusive `(start, end)` bounds from
`SegmentCache.contained()` rather than reading the one-integer-per-window
`seg_id`, which cannot express a window being in two places at once.

**What is *not* reproduced here is the outer averaging.** The scorer averages
over segments, then trajectories, then platforms, equally. This returns the mean
over the segments present in one batch -- the innermost average only, exactly as
the AVE term does. Matching the outer three is the sampler's job (compose.py).
"""
from __future__ import annotations

import torch

#: Floor under the mean squared error before its square root is taken. The
#: derivative of sqrt is infinite at the origin, and a perfectly aligned segment
#: is not a hypothetical -- it is what a well-trained model produces on a short
#: segment.
#:
#: It is a **clamp, not an added epsilon**, and that distinction is worth 5e-10
#: of exactness. Adding eps inside the radicand perturbs *every* segment:
#: sqrt(x + 1e-12) exceeds sqrt(x) by 5e-10 when x is 1e-6, which is exactly the
#: disagreement with the official scorer that this term used to carry. Clamping
#: leaves every realistic segment untouched (1e-24 is a 1e-12 m error, far below
#: anything a float can measure over 20 m) and the clamp's own backward pass
#: returns a zero gradient below the floor instead of an infinite one.
EPS = 1e-24

#: Where the term runs, regardless of where the model lives. See point 3 above.
COMPUTE_DEVICE = torch.device("cpu")
COMPUTE_DTYPE = torch.float64


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """`[x, y, z, w]` (scalar-last) to a body-to-world rotation matrix, `(..., 3, 3)`.

    Transcribed from the scorer's `_quat_to_R`, **including the normalisation**,
    which is not cosmetic. The shipped quaternions are unit to within their
    storage precision, not exactly unit, so skipping the division moves the
    result in the seventh digit -- measured, it was the whole of a 2.7e-07
    disagreement with the scorer that otherwise had no explanation. A zero
    quaternion is mapped to identity exactly as the scorer maps it, rather than
    to NaN.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = torch.sqrt(x * x + y * y + z * z + w * w)
    n = torch.where(n == 0, torch.ones_like(n), n)
    x, y, z, w = x / n, y / n, z / n, w / n
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return r.reshape(*q.shape[:-1], 3, 3)


def _det3(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a batch of 3x3 matrices, by cofactor expansion.

    `torch.linalg.det` is 340x slower than this on MPS for a batch this shape,
    and it, not the singular value decomposition, is what made the first
    implementation of this term expensive.
    """
    return (m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
            - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
            + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]))


def umeyama_align(P: torch.Tensor, Q: torch.Tensor,
                  point_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched least-squares `SE(3)` alignment of `P` onto `Q`, padding-aware.

    `P`, `Q` are `(N, M, 3)` and `point_mask` is `(N, M)`; padded points are
    excluded from both the centroids and the cross-covariance, so a short
    segment padded out to the batch's longest gives exactly the same answer it
    would give alone.

    **Returned detached**, for the two reasons in this module's docstring: the
    gradient through them is zero by the envelope theorem, and keeping the
    decomposition out of the graph is what stops a zero prediction producing NaN.
    """
    m = point_mask.to(P.dtype).unsqueeze(-1)             # (N, M, 1)
    n = m.sum(dim=1).clamp(min=1.0)                      # (N, 1)

    muP = (P * m).sum(dim=1) / n                         # (N, 3)
    muQ = (Q * m).sum(dim=1) / n
    Pc = (P - muP.unsqueeze(1)) * m                      # padding contributes zero
    Qc = (Q - muQ.unsqueeze(1)) * m

    H = torch.einsum("nmi,nmj->nij", Pc, Qc)
    U, _, Vt = torch.linalg.svd(H)

    # d = sign(det(Vt.T @ U.T)) in the scorer. det(A.T) = det(A) and the
    # determinant is multiplicative, so this is det(Vt) * det(U) -- the same
    # number without forming the product. `sign` is used rather than a
    # comparison so a genuinely singular H behaves as the scorer's np.sign does.
    d = torch.sign(_det3(Vt) * _det3(U))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))

    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    t = muQ - torch.einsum("nij,nj->ni", R, muP)
    return R.detach(), t.detach()


def segment_rms(v_pred: torch.Tensor, q_gt: torch.Tensor, p_gt: torch.Tensor,
                dt: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor,
                *, eps: float = EPS) -> torch.Tensor:
    """Per-segment aligned RMS position error, `(N,)`, one entry per real segment.

    Shapes are the batch's: `v_pred (B,K,3)`, `q_gt (B,K,4)`, `p_gt (B,K,3)`,
    `dt (B,K)`, and `seg_start`/`seg_end` `(B,S)` chunk-relative inclusive bounds
    with `-1` marking a padded slot. Returns an empty tensor when the batch
    contains no whole segment, which is the common case on long-trajectory
    platforms at small `chunk_len` and is not an error.
    """
    valid = seg_start >= 0
    if not bool(valid.any()):
        return v_pred.new_zeros((0,))

    b_idx, slot = valid.nonzero(as_tuple=True)
    starts = seg_start[b_idx, slot]
    ends = seg_end[b_idx, slot]
    lengths = ends - starts + 1
    span = torch.arange(int(lengths.max()), device=v_pred.device)
    point_mask = span.unsqueeze(0) < lengths.unsqueeze(1)                 # (N, M)
    idx = (starts.unsqueeze(1) + span.unsqueeze(0)).clamp(max=v_pred.shape[1] - 1)

    # World-frame displacement per window, gathered per segment and integrated
    # **inside** the segment, exactly as the scorer does.
    #
    # A whole-chunk prefix sum with the segment's own offset subtracted would be
    # the same quantity on paper, and -- measured on every val segment, with
    # prefixes reaching 155 m before a 20 m segment is differenced back out --
    # it is also the same to 1.9e-14 in float64, so the cancellation this form
    # avoids turns out not to bite. It is kept anyway because it is literally
    # what the scorer computes, at the same cost.
    disp = torch.einsum("bkij,bkj->bki", quat_to_matrix(q_gt), v_pred) * dt.unsqueeze(-1)
    m3 = point_mask.unsqueeze(-1).to(disp.dtype)
    P = torch.cumsum(disp[b_idx.unsqueeze(1), idx] * m3, dim=1)
    Q = p_gt[b_idx.unsqueeze(1), idx]

    R, t = umeyama_align(P, Q, point_mask)
    err = torch.einsum("nij,nmj->nmi", R, P) + t.unsqueeze(1) - Q
    sq = (err * err).sum(-1) * point_mask.to(err.dtype)                   # (N, M)
    n = point_mask.to(err.dtype).sum(dim=1).clamp(min=1.0)
    return torch.sqrt((sq.sum(dim=1) / n).clamp(min=eps))


def masked_ate20(v_pred: torch.Tensor, q_gt: torch.Tensor, p_gt: torch.Tensor,
                 dt: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor,
                 *, device: torch.device = COMPUTE_DEVICE,
                 dtype: torch.dtype = COMPUTE_DTYPE,
                 eps: float = EPS) -> torch.Tensor:
    """Mean over the batch's whole segments of the per-segment RMS, as a scalar.

    **The outer square root sits inside the mean and the nesting cannot be
    flattened.** A global RMS over all points would dilute a short, badly wrong
    segment with a long clean one; the metric does not, and neither does this.

    Everything is computed on `device` in `dtype` -- CPU float64 by default,
    which is both faster and exact. The result is returned in the caller's dtype
    and on the caller's device so it can be summed straight into a loss.
    """
    out_device, out_dtype = v_pred.device, v_pred.dtype

    # Device first, dtype second, and never in one call. `.to(device=cpu,
    # dtype=float64)` on an MPS tensor raises -- MPS has no float64, so the
    # combined form is attempted as a cast *before* the move. Landing on the CPU
    # first makes the same conversion legal.
    def bring(t: torch.Tensor, d: torch.dtype | None = None) -> torch.Tensor:
        t = t.to(device=device)
        return t if d is None else t.to(dtype=d)

    per_segment = segment_rms(
        bring(v_pred, dtype), bring(q_gt, dtype), bring(p_gt, dtype),
        bring(dt, dtype), bring(seg_start), bring(seg_end), eps=eps)
    if per_segment.numel() == 0:
        return v_pred.new_zeros(())
    # Cast on the CPU, then move -- the same two-step rule as `bring`, and it
    # matters here for the *backward* pass, which walks this line in reverse and
    # would otherwise ask MPS for a float64 tensor.
    return per_segment.mean().to(dtype=out_dtype).to(device=out_device)


def ate20_from_batch(v_pred: torch.Tensor, batch, **kwargs) -> torch.Tensor:
    """`masked_ate20` with the batch keys unpacked -- what the loss bundle calls."""
    for key in ("seg_start", "seg_end"):
        if key not in batch:
            raise KeyError(
                f"the ATE20 loss term needs {key!r}, which the batch does not carry. "
                f"Build the dataset with segments=load_segments(split); on test "
                f"there are no segments at all, so the term cannot run there.")
    return masked_ate20(v_pred, batch["q_gt"], batch["p_gt"], batch["dt"],
                        batch["seg_start"], batch["seg_end"], **kwargs)
