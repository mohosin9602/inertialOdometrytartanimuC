"""Tier 0 — assemble the C-channel model input. PipelinePlan.md §2.4, §4.2.

`prep.orientation` (attitude estimation) and `prep.features` (channel assembly)
are deliberately separate: the learned-attitude experiment swaps only the first,
the input-representation ablation only the second.

| input_repr     | C | channels                                                     |
| -------------- | - | ------------------------------------------------------------ |
| body           | 6 | ax ay az (gravity included), gx gy gz                        |
| body_grav      | 9 | the above + ux uy uz, gravity direction in body coordinates   |
| aligned        | 6 | acc/gyro rotated into a tilt-corrected local frame           |
| aligned_grav   | 9 | aligned + the body-frame tilt the rotation just removed        |

**All four are built.** The `aligned` pair rotate the raw signal into a frame
whose vertical axis is the estimated gravity direction, so the network is handed
a signal that has already been straightened instead of having to learn which way
is up. `CLAUDE.md` lever 3 and `DIAGNOSIS.md` D11.

**The rotation comes from the complementary filter, never from `quat`.** An
earlier version of this docstring said the `aligned` modes take ground-truth
attitude. Built that way they would be illegal on test, which carries none --
so `align_rotation` reads the same distribution-free down-direction estimate
`body_grav` already appends, and the `quat` argument is now unused by every
representation.

**`aligned_grav`'s last three channels are the tilt in BODY coordinates, not a
residual.** The residual would be a constant: the rotation maps the estimated
gravity onto +z at every frame by construction, so what is left is exactly
`(0, 0, 1)` and carries no information at all. What the alignment actually
*destroys* is the tilt itself -- how the box is bolted on and how the platform
is leaning -- and that is what the three extra channels put back.

**The gravity channels are per-frame** — a genuine time series produced by the
complementary filter in `prep.orientation`, not one constant broadcast over the
window. That matters: a vehicle cresting a hill changes tilt *within* a second,
and a constant would erase exactly that.
"""
from __future__ import annotations

import numpy as np

N_CHANNELS = {"body": 6, "body_grav": 9, "aligned": 6, "aligned_grav": 9}
EXTRA_SCALARS = ("acc_norm_minus_g", "gyro_norm")

#: Which representations need `prep.orientation`'s cached gravity directions.
#: `aligned` is in the list even though it appends nothing: it needs the estimate
#: to build its rotation.
NEEDS_GRAVITY = ("body_grav", "aligned", "aligned_grav")

#: Which representations predict in the gravity-aligned frame, and therefore need
#: their velocity output rotated back into the body frame before anything reads
#: it (`models.base.forward_batch`). Getting this list wrong is silent: the
#: submission would be a correct prediction expressed in the wrong frame.
ALIGNED = ("aligned", "aligned_grav")


def is_aligned(input_repr: str) -> bool:
    return input_repr in ALIGNED


def align_rotation(gravity: np.ndarray) -> np.ndarray:
    """`(F, 3)` unit down-directions in body coordinates -> `(F, 3, 3)` rotations.

    Each matrix is the **minimum-angle** rotation taking that frame's estimated
    gravity direction onto `+z`. Minimum-angle is the whole design: it removes
    roll and pitch and touches heading not at all, so nothing absolute -- which
    an accelerometer and a gyroscope cannot know -- is ever invented, and no
    heading drift can enter through the back door.

    Rodrigues' formula for the rotation carrying `a` onto `b`, with
    `v = a x b` and `c = a . b`:

        R = I + [v]x + [v]x^2 / (1 + c)

    which for `b = (0, 0, 1)` gives `v = (g_y, -g_x, 0)` and `c = g_z`. The
    formula degenerates only when `c = -1`, i.e. gravity pointing at exactly
    `-z`, where the rotation is a half turn about any horizontal axis; `+x` is
    chosen there. That branch is not hypothetical -- dog and drone both have
    trajectories past 90 degrees of tilt.
    """
    g = np.asarray(gravity, np.float64)
    g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    f = g.shape[0]
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]

    # Axis and angle of the rotation carrying g onto +z:
    #     axis = g x (0,0,1) = (gy, -gx, 0),   cos(theta) = gz
    # The axis is NORMALISED before use. Writing Rodrigues with the unit axis and
    # (sin, 1-cos) keeps every term of order one; the textbook `1/(1+c)` form is
    # algebraically identical but divides by something that goes to zero as
    # gravity approaches -z, and in float32 that cost 0.06 degrees of alignment
    # on the worst row of a 20,000-row stress test. This form costs nothing.
    s = np.hypot(gx, gy)                       # = |g x z|
    ok = s > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, s, 1.0), 0.0)
    nx, ny = gy * inv, -gx * inv               # unit axis; nz is identically 0

    K = np.zeros((f, 3, 3), np.float64)        # [n]x for n = (nx, ny, 0)
    K[:, 0, 2] = ny
    K[:, 1, 2] = -nx
    K[:, 2, 0] = -ny
    K[:, 2, 1] = nx

    R = np.broadcast_to(np.eye(3), (f, 3, 3)).copy()
    R += s[:, None, None] * K + (1.0 - gz)[:, None, None] * (K @ K)

    if (~ok).any():
        # g is exactly +z (already level: the identity, which the formula above
        # already gives) or exactly -z (upside down: a half turn about any
        # horizontal axis; +x is as good as any).
        flip = np.diag([1.0, -1.0, -1.0])
        upside = (~ok) & (gz < 0)
        R[~ok] = np.eye(3)
        if upside.any():
            R[upside] = flip
    return R.astype(np.float32)


def n_channels(input_repr: str, extra_scalars: tuple | list = ()) -> int:
    if input_repr not in N_CHANNELS:
        raise KeyError(f"unknown input_repr {input_repr!r}; have {sorted(N_CHANNELS)}")
    return N_CHANNELS[input_repr] + len(extra_scalars)


def needs_gravity(input_repr: str) -> bool:
    return input_repr in NEEDS_GRAVITY


def assemble(imu: np.ndarray, input_repr: str = "body",
             quat: np.ndarray | None = None,
             extra_scalars: tuple | list = (),
             gravity: np.ndarray | None = None) -> np.ndarray:
    """`(F, 6)` raw IMU frames -> `(F, C)` model input channels, float32.

    `gravity` is the `(F, 3)` per-frame down-direction from `prep.orientation`,
    required by every representation whose name ends in `_grav`.
    `quat` is ground-truth attitude and is used by the deferred `aligned` modes
    only — it exists on train and val alone, so nothing on a submission path may
    depend on it.
    """
    x = np.asarray(imu, dtype=np.float32)

    if input_repr == "body":
        out = x
    elif input_repr == "body_grav":
        if gravity is None:
            raise ValueError(
                "input_repr='body_grav' needs the per-frame gravity directions. "
                "Build them once with:\n"
                "    mySolution/.venv/bin/python -m src.prep.orientation")
        g = np.asarray(gravity, np.float32)
        if g.shape[0] != x.shape[0]:
            raise ValueError(f"gravity has {g.shape[0]} frames but the IMU has "
                             f"{x.shape[0]}")
        out = np.concatenate([x, g], axis=1)
    elif input_repr in ALIGNED:
        if gravity is None:
            raise ValueError(
                f"input_repr={input_repr!r} needs the per-frame gravity directions to "
                f"build its rotation. Build them once with:\n"
                f"    mySolution/.venv/bin/python -m src.prep.orientation")
        g = np.asarray(gravity, np.float32)
        if g.shape[0] != x.shape[0]:
            raise ValueError(f"gravity has {g.shape[0]} frames but the IMU has "
                             f"{x.shape[0]}")
        rot = align_rotation(g)
        # Per frame, not per window. A vehicle cresting a hill changes tilt
        # inside a single second, and a per-window rotation would smear that
        # change across the whole window instead of removing it.
        acc = np.einsum("fij,fj->fi", rot, x[:, 0:3])
        gyr = np.einsum("fij,fj->fi", rot, x[:, 3:6])
        out = (np.concatenate([acc, gyr], axis=1) if input_repr == "aligned"
               else np.concatenate([acc, gyr, g], axis=1))
    else:
        raise KeyError(f"unknown input_repr {input_repr!r}")

    if extra_scalars:
        cols = [out]
        if "acc_norm_minus_g" in extra_scalars:
            cols.append(np.linalg.norm(x[:, :3], axis=1, keepdims=True) - 9.81)
        if "gyro_norm" in extra_scalars:
            cols.append(np.linalg.norm(x[:, 3:6], axis=1, keepdims=True))
        out = np.concatenate(cols, axis=1)
    return out
