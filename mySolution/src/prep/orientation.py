"""Tier 0 -- estimate which way is down, from the IMU alone. GravityDirectionPlan.md.

    mySolution/.venv/bin/python -m src.prep.orientation             # all three splits
    mySolution/.venv/bin/python -m src.prep.orientation --verify    # check it against truth

**What this is for, in one paragraph.** The accelerometer measures gravity and
real acceleration added together, and there is no way to tell them apart from a
single reading. A car accelerating gently on level ground and a car parked
nose-up on a ramp produce accelerometer readings that differ by about two
percent, yet one is moving and the other is not. What separates them is history:
the gyroscope says the parked car has not rotated since the recording began, so
its tilt is old news rather than fresh acceleration. This module runs that
history forward and writes down, for every frame, a unit vector pointing the way
gravity does -- expressed in the body's own coordinates. Those three numbers
become input channels 7, 8 and 9 (`input_repr = "body_grav"`).

**Why a direction and not a full orientation.** A full orientation also carries
heading -- which way is north -- and nothing in this project needs heading. Both
the input and the target live in the body frame, so a recording rotated about
the vertical axis is the same sample. Tracking only the down-direction means two
numbers of real state instead of three, no quaternion algebra, and no drift in
the one component that could never be corrected anyway.

**The filter, in plain terms.** Two sources disagree and we take a weighted
average of them at every frame:

* The **gyroscope** says how the body just turned, so it can carry last frame's
  answer forward. It is smooth and trustworthy over a second, and slowly wrong
  over a minute.
* The **accelerometer** says where down is *if the body is not accelerating*.
  It never drifts, but it is wrong exactly when the platform manoeuvres.

So: propagate with the gyroscope, then nudge a little way back toward the
accelerometer. The nudge is scaled by how believable the accelerometer looks --
a reading whose length is far from 9.81 m/s^2 is contaminated by real
acceleration and is trusted less. This is the standard complementary filter,
written for a direction instead of a quaternion.

**The sign convention is measured, not assumed.** An accelerometer at rest reads
*specific force*, which points away from gravity -- a resting sensor reads +9.81
upward, not downward. Whether the cached vector points down or up is therefore a
convention, and getting it backwards is silent and catastrophic: nothing crashes,
the model just receives an inverted channel forever. `verify()` settles it
against the ground-truth quaternions that ship with train and val, and the
answer is recorded in the cache's meta.json.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..paths import CACHE_V, PLATFORMS, SPLITS
from ..utils import env_record, write_json
from .cache import load_cache

#: Standard gravity, m/s^2. Used only to judge how much to trust a reading.
G = 9.80665

#: How hard the accelerometer pulls the estimate back per frame. **Measured**,
#: not chosen: swept over {0.001 .. 0.1} against ground-truth attitude on 16 val
#: trajectories, four per platform, scored by the median angle to truth and
#: macro-averaged over platforms the way the competition metric is.
#: 0.003 wins at 1.84 degrees. The trade-off is real and visible in the sweep --
#: car and dog prefer 0.001 (their tilt is genuinely steady, so trusting the
#: gyroscope pays), drone prefers 0.1 (its vibration makes gyro integration
#: drift fastest). 0.003 is the compromise the macro-average picks.
#: Reproduce: ../.claude/agentTests/2026-08-31_m5-model/tune_gravity_alpha.py
DEFAULT_ALPHA = 0.003

#: How far |acc| may stray from 9.81 before the reading is discounted, m/s^2.
#: The trust factor is a Gaussian bump, so a reading this far off contributes
#: about a third as much as a clean one. The same sweep shows the result is
#: insensitive to it (1.84 vs 1.94 degrees over 1.0 .. 5.0), which says the
#: small alpha is already doing the work of rejecting contaminated readings.
DEFAULT_TOLERANCE = 5.0

#: Which end of the world's vertical axis the cached vector points along, and
#: therefore the sign that makes the filter and the ground truth agree.
#: **Measured, never assumed**: at +1.0 the filter lands 1.8 degrees from truth,
#: at -1.0 it lands 178 degrees away. An accelerometer at rest reports specific
#: force, so which way it points relative to a dataset's world axis is a
#: convention, and getting it backwards is silent -- nothing crashes, the model
#: simply receives an inverted channel forever. `verify()` re-checks it on
#: demand and reports both options side by side.
DOWN_SIGN = +1.0

#: The estimators `prep.gravity.estimator` can name (M19). "cf" is the causal
#: filter above, which every run through M18 used; an absent key means "cf", so
#: no earlier config hash and no earlier cache file moves. The other two look
#: both ways in time, which the offline-smoothing category allows (the
#: 2026-09-07 ruling). Measured against ground truth, median tilt error in
#: degrees, val: `.claude/agentTests/2026-09-18_gravity-estimators/`
#:
#:     estimator   car   dog   drone  human   macro val  macro train
#:     cf          0.82  1.58  6.57   1.52    2.62       3.01
#:     cf_fb       0.52  1.25  5.00   1.23    2.00       2.31
#:     box (5 s)   0.33  0.81  5.40   1.15    1.92       2.11
#:     rts         0.27  0.83  5.39   0.86    1.84       --
#:
#: `rts` (lever 5) is a Kalman filter plus a backward RTS pass, with the gyro's
#: trust chosen per recording by maximum likelihood -- tune_rts.out and
#: tune_rts_kappa.out in the same folder. With ONE fixed long gyro memory it
#: scores drone 10.55 deg (p90 71): the drone gyro-frame defect (finding 21).
ESTIMATORS = ("cf", "cf_fb", "box", "rts")

#: Half-width of the two-sided window, seconds. 5 won the platform-blind sweep
#: over {1, 2, 5, 10, 20}: longer windows help car, dog and human and hurt drone.
DEFAULT_BOX_SECONDS = 5.0


def gravity_direction(imu: np.ndarray, dt: float,
                      alpha: float = DEFAULT_ALPHA,
                      tolerance: float = DEFAULT_TOLERANCE,
                      down_sign: float = DOWN_SIGN) -> np.ndarray:
    """One trajectory's `(N, 6)` IMU -> `(N, 3)` unit vectors pointing down, body frame.

    The loop is over frames because each answer depends on the one before it;
    that is what "carrying history forward" means and it cannot be vectorised
    away. Everything inside the loop is plain float arithmetic on three numbers,
    which is far faster than building tiny numpy arrays 200 times a second.
    """
    imu = np.ascontiguousarray(imu, dtype=np.float64)
    n = imu.shape[0]
    out = np.zeros((n, 3), np.float64)
    if n == 0:
        return out.astype(np.float32)

    acc = imu[:, 0:3]
    gyro = imu[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)
    # How much to believe each accelerometer reading: 1.0 when its length is
    # exactly gravity, tailing off smoothly as real acceleration contaminates it.
    trust = np.exp(-(((acc_norm - G) / tolerance) ** 2))
    # Guard against a genuinely zero reading (free fall, or a dropped sample).
    safe = np.maximum(acc_norm, 1e-9)
    acc_unit = (acc / safe[:, None]) * down_sign

    # Seed from the first reading. It may be wrong if the recording starts mid
    # manoeuvre, but the filter converges within a fraction of a second.
    gx, gy, gz = acc_unit[0]

    gyro_list = gyro.tolist()          # python floats: ~3x faster than numpy indexing
    acc_list = acc_unit.tolist()
    trust_list = trust.tolist()

    for i in range(n):
        wx, wy, wz = gyro_list[i]
        # A world-fixed direction, seen from a body rotating at angular velocity
        # w, drifts as  dg/dt = -(w x g).  One first-order step of that is all a
        # 5 ms frame needs; the error is O(dt^2) and washes out in the blend.
        cx = wy * gz - wz * gy
        cy = wz * gx - wx * gz
        cz = wx * gy - wy * gx
        gx -= cx * dt
        gy -= cy * dt
        gz -= cz * dt

        # Pull a little way back toward what the accelerometer claims, in
        # proportion to how believable this particular reading is.
        a = alpha * trust_list[i]
        ax, ay, az = acc_list[i]
        gx += a * (ax - gx)
        gy += a * (ay - gy)
        gz += a * (az - gz)

        # Renormalise: both the propagation and the blend shrink the length
        # slightly, and only the direction carries meaning.
        length = (gx * gx + gy * gy + gz * gz) ** 0.5
        if length > 1e-12:
            gx, gy, gz = gx / length, gy / length, gz / length

        out[i, 0] = gx
        out[i, 1] = gy
        out[i, 2] = gz

    return out.astype(np.float32)


# ------------------------------------------------------------ two-sided estimators
#
# Both look forward in time as well as back, so they belong to the offline
# smoothing category the submission already declares. The causal filter above
# stays the default, and the causal track keeps it.
#
# Quaternions are [x, y, z, w], matching the rest of the project. These helpers
# are vectorised over frames; `.claude/agentTests/2026-09-18_gravity-estimators/
# check_helpers.py` checks each one against explicit rotation matrices.

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack((aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def _conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _expmap(v: np.ndarray) -> np.ndarray:
    """Rotation vectors -> unit quaternions."""
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    s = np.where(ang > 1e-8, np.sin(0.5 * ang) / np.maximum(ang, 1e-300),
                 0.5 - ang ** 2 / 48.0)
    return np.concatenate((v * s, np.cos(0.5 * ang)), axis=-1)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) v, row by row."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _cumprod(q: np.ndarray) -> np.ndarray:
    """Inclusive left-to-right quaternion product, as a log-depth doubling scan.

    A frame-by-frame loop over a 774,081-frame dog recording is seconds of
    pure Python; the scan is twenty vectorised passes.
    """
    c, d = q.copy(), 1
    while d < len(c):
        nxt = c.copy()
        nxt[d:] = _qmul(c[:-d], c[d:])
        nxt /= np.linalg.norm(nxt, axis=1, keepdims=True)
        c, d = nxt, d * 2
    return c


def attitude_from_gyro(gyro: np.ndarray, dt: float) -> np.ndarray:
    """`(N, 3)` body rates -> `(N, 4)` quaternions rotating body-i into body-0.

    `Q[i+1] = Q[i] (x) Exp(w_i dt)`. Heading drifts freely -- nothing here
    needs it. Only the rotation *between* two frames is ever used, and that
    drifts only over the time separating them.
    """
    Q = np.empty((len(gyro), 4))
    if len(gyro) == 0:
        return Q
    Q[0] = (0.0, 0.0, 0.0, 1.0)
    if len(gyro) > 1:
        Q[1:] = _cumprod(_expmap(np.asarray(gyro[:-1], np.float64) * dt))
    return Q


def gravity_forward_backward(imu: np.ndarray, dt: float,
                             alpha: float = DEFAULT_ALPHA,
                             tolerance: float = DEFAULT_TOLERANCE,
                             down_sign: float = DOWN_SIGN) -> np.ndarray:
    """The causal filter run forward, then backward in time, and averaged.

    Running a recording backwards turns every rotation the other way, so the
    backward pass negates the gyroscope; the accelerometer is unchanged. Each
    pass is wrong where it has not converged yet -- the forward one at the start,
    the backward one at the end -- and the average is right at both.
    """
    x = np.asarray(imu, np.float64)
    fwd = gravity_direction(x, dt, alpha, tolerance, down_sign).astype(np.float64)
    rev = x[::-1].copy()
    rev[:, 3:6] *= -1.0
    bwd = gravity_direction(rev, dt, alpha, tolerance, down_sign).astype(np.float64)[::-1]
    g = fwd + bwd
    return (g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def gravity_box(imu: np.ndarray, dt: float,
                half_window_s: float = DEFAULT_BOX_SECONDS,
                down_sign: float = DOWN_SIGN) -> np.ndarray:
    """Gravity as the specific force averaged over +-`half_window_s` seconds.

    The accelerometer reads gravity plus the platform's own acceleration. Over
    ten seconds the platform's speeding up and slowing down cancel -- the mean
    acceleration over a window is (v_end - v_start) / duration, small once the
    window is long -- and what is left points along gravity. The readings are
    first carried into one fixed frame by integrating the gyroscope, so a body
    that turns during the window is averaged correctly; the average is then
    carried back into the current frame. Only rotations across the window are
    used, so gyroscope drift enters over ten seconds, never over the flight.
    """
    x = np.asarray(imu, np.float64)
    n = len(x)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    Q = attitude_from_gyro(x[:, 3:6], dt)
    P = np.concatenate((np.zeros((1, 3)), np.cumsum(_rotate(Q, x[:, 0:3]), axis=0)))
    w = int(round(half_window_s / dt))
    i = np.arange(n)
    g = _rotate(_conj(Q), P[np.minimum(i + w + 1, n)] - P[np.maximum(i - w, 0)])
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return (down_sign * g).astype(np.float32)


#: The Kalman smoother's settings (M19, lever 5). `block` frames per step (20 Hz);
#: `sigma_acc` and `kappa` are tuned against ground truth on train and confirmed
#: on val, like the filter's alpha --
#: `.claude/agentTests/2026-09-18_gravity-estimators/tune_rts.py`. `sigma_gyro`
#: is the fixed value when `adaptive` is 0; when it is 1 the gyroscope's trust is
#: chosen PER RECORDING, by maximum likelihood over `RTS_GYRO_GRID` (see
#: `gravity_rts`), which is what a mis-framed drone gyroscope needs.
DEFAULT_RTS = {"block": 10, "sigma_gyro": 0.001, "sigma_bias": 1e-4,
               "sigma_acc": 0.03, "kappa": 100.0, "adaptive": 1}
RTS_GYRO_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


def _skew(v: np.ndarray) -> np.ndarray:
    """`(..., 3)` -> `(..., 3, 3)` cross-product matrices."""
    o = np.zeros(v.shape[:-1])
    return np.stack((np.stack((o, -v[..., 2], v[..., 1]), -1),
                     np.stack((v[..., 2], o, -v[..., 0]), -1),
                     np.stack((-v[..., 1], v[..., 0], o), -1)), -2)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((
        np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def _rts_prepare(imu: np.ndarray, dt: float, block: int) -> dict:
    """Everything the filter needs that does not depend on its settings."""
    x = np.asarray(imu, np.float64)[:, :6]
    n = len(x)
    D = max(1, int(block))
    K = -(-n // D)
    if K * D > n:                                  # pad the last block by repetition
        x = np.concatenate((x, np.repeat(x[-1:], K * D - n, axis=0)))
    acc = x[:, 0:3].reshape(K, D, 3)
    inc = _expmap(x[:, 3:6] * dt).reshape(K, D, 4)
    # Q[:, j] rotates frame j of each block into the block's first frame.
    Q = np.empty((K, D + 1, 4))
    Q[:, 0] = (0.0, 0.0, 0.0, 1.0)
    for j in range(D):
        Q[:, j + 1] = _qmul(Q[:, j], inc[:, j])
    Q[:, 1:] /= np.linalg.norm(Q[:, 1:], axis=-1, keepdims=True)
    carried = _rotate(Q[:, :D], acc)               # (K, D, 3) in block-start frame
    mean = carried.mean(axis=1)
    norm = np.linalg.norm(mean, axis=1)
    spread = ((carried - mean[:, None]) ** 2).sum(-1).mean(1)
    return {"n": n, "K": K, "D": D, "T": D * dt, "Q": Q,
            "z": mean / np.maximum(norm, 1e-12)[:, None],
            "dyn": ((norm - G) ** 2 + spread) / G ** 2,
            "C": _quat_to_mat(Q[:, D])}            # block k+1 frame -> block k frame


def _rts_forward(p: dict, sigma_gyro, sigma_bias, sigma_acc, kappa, store: bool):
    """The Kalman filter, run for several settings at once (the leading axis).

    Returns the negative log-likelihood of the accelerometer under each setting
    -- the sum over blocks of log|S| + y'S^-1 y, the standard way to let the
    data pick a filter's noise levels -- and, when `store` (one setting only),
    everything the RTS pass needs.
    """
    sg, sb, sa, ka = (np.atleast_1d(np.asarray(v, np.float64))
                      for v in (sigma_gyro, sigma_bias, sigma_acc, kappa))
    Gn = max(len(sg), len(sb), len(sa), len(ka))
    sg, sb, sa, ka = (np.broadcast_to(v, (Gn,)) for v in (sg, sb, sa, ka))
    K, T, z, C = p["K"], p["T"], p["z"], p["C"]
    I3 = np.eye(3)
    g = np.repeat(z[:1], Gn, axis=0)
    b = np.zeros((Gn, 3))
    P = np.repeat(np.diag([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])[None], Gn, axis=0)
    nll = np.zeros(Gn)
    if store:
        xs_f, Ps_f = np.empty((K, 6)), np.empty((K, 6, 6))
        xs_p, Ps_p, Fs = np.empty((K, 6)), np.empty((K, 6, 6)), np.empty((K, 6, 6))
    qg = (sg ** 2 * T)[:, None, None]
    qb = (sb ** 2 * T)[:, None, None] * I3
    for k in range(K):
        if k > 0:
            # predict across block k-1: the gyro, bias removed, turns g
            Cb = C[k - 1] @ (I3 - _skew(b * T))
            gn = np.einsum("gji,gj->gi", Cb, g)
            gn /= np.linalg.norm(gn, axis=1, keepdims=True)
            F = np.zeros((Gn, 6, 6))
            F[:, :3, :3] = np.transpose(Cb, (0, 2, 1))
            F[:, :3, 3:] = -_skew(gn) * T
            F[:, 3:, 3:] = I3
            Qn = np.zeros((Gn, 6, 6))
            Qn[:, :3, :3] = qg * (I3 - np.einsum("gi,gj->gij", gn, gn)) + 1e-12 * I3
            Qn[:, 3:, 3:] = qb
            P = F @ P @ np.transpose(F, (0, 2, 1)) + Qn
            g = gn
            if store:
                Fs[k - 1] = F[0]
        if store:
            xs_p[k, :3], xs_p[k, 3:], Ps_p[k] = g[0], b[0], P[0]
        # update with block k's accelerometer, trusted less when it shows motion
        S = P[:, :3, :3] + (sa ** 2 + ka * p["dyn"][k])[:, None, None] * I3
        y = z[k][None] - g
        Sy = np.linalg.solve(S, y[..., None])[..., 0]
        nll += np.log(np.linalg.det(S)) + np.sum(y * Sy, axis=1)
        Kg = np.transpose(np.linalg.solve(S, P[:, :3, :]), (0, 2, 1))   # P H' S^-1
        upd = np.einsum("gij,gj->gi", Kg, y)
        g = g + upd[:, :3]
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        b = b + upd[:, 3:]
        P = P - Kg @ P[:, :3, :]
        P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
        if store:
            xs_f[k, :3], xs_f[k, 3:], Ps_f[k] = g[0], b[0], P[0]
    if not store:
        return nll, None
    return nll, (xs_f, Ps_f, xs_p, Ps_p, Fs)


def gravity_rts(imu: np.ndarray, dt: float, block: int = DEFAULT_RTS["block"],
                sigma_gyro: float = DEFAULT_RTS["sigma_gyro"],
                sigma_bias: float = DEFAULT_RTS["sigma_bias"],
                sigma_acc: float = DEFAULT_RTS["sigma_acc"],
                kappa: float = DEFAULT_RTS["kappa"],
                adaptive: int = DEFAULT_RTS["adaptive"],
                down_sign: float = DOWN_SIGN) -> np.ndarray:
    """Gravity from a Kalman filter run forward, then an RTS smoother run backward.

    The research doc's "RTS smoothing on an attitude EKF", sized for this job.

    *State*, every `block` frames: the down-direction `g` (a unit 3-vector in
    body coordinates) and the gyroscope's bias `b`. *Prediction*: the gyroscope,
    bias removed, turns `g` across the block, exactly as a world-fixed vector
    turns when seen from a rotating body. *Measurement*: the block's
    accelerometer readings, carried to the block's first frame by the same
    rotation and averaged, point along `g` -- trusted LESS when the block shows
    the platform accelerating (its mean length far from 9.81, or its readings
    spread out): `R = sigma_acc^2 + kappa * dyn`. *Smoothing*: the RTS pass runs
    backward and corrects every state with everything that came after it, so the
    start of a recording is as good as its middle. *Output*: every frame between
    two smoothed states is carried forward from the one before and back from the
    one after, and the two blended, so the channel is continuous.

    *How far to trust the gyroscope* (`adaptive`): on most drone flights the
    gyroscope is not in the accelerometer's frame (CLAUDE.md finding 21), so
    the long gyro memory that is right for a car is ruinous there. With
    `adaptive=1` the filter is run for every value in `RTS_GYRO_GRID` at once and
    the one under which the recording's own accelerometer is most likely wins --
    maximum likelihood on the innovations, per recording. No label, no ground
    truth: a healthy gyroscope earns a long memory, a mis-framed one a short one.
    """
    if len(imu) == 0:
        return np.zeros((0, 3), np.float32)
    p = _rts_prepare(imu, dt, block)
    sg = sigma_gyro
    if adaptive:
        grid = np.asarray(RTS_GYRO_GRID)
        nll, _ = _rts_forward(p, grid, sigma_bias, sigma_acc, kappa, store=False)
        sg = float(grid[int(np.argmin(nll))])
    _, (xs_f, Ps_f, xs_p, Ps_p, Fs) = _rts_forward(p, sg, sigma_bias, sigma_acc,
                                                     kappa, store=True)
    K, D, Q, C, n = p["K"], p["D"], p["Q"], p["C"], p["n"]
    xs = xs_f.copy()
    Ps = Ps_f[-1].copy()
    for k in range(K - 2, -1, -1):
        Gk = np.linalg.solve(Ps_p[k + 1], Fs[k] @ Ps_f[k]).T   # P_f F' P_p^-1
        xs[k] = xs_f[k] + Gk @ (xs[k + 1] - xs_p[k + 1])
        xs[k, :3] /= np.linalg.norm(xs[k, :3])
        Ps = Ps_f[k] + Gk @ (Ps - Ps_p[k + 1]) @ Gk.T

    # Per frame: carried forward from the state at the block's start, back from
    # the state at the next block's start, blended linearly across the block.
    gs = xs[:, :3]
    fwd = _rotate(_conj(Q[:, :D]), np.repeat(gs[:, None], D, axis=1))
    nxt = np.concatenate((gs[1:], gs[-1:]))        # the last block has no successor
    at_end = np.einsum("kij,kj->ki", C, nxt)       # next state, in this block's first frame
    bwd = _rotate(_conj(Q[:, :D]), np.repeat(at_end[:, None], D, axis=1))
    w = (np.arange(D) / D)[None, :, None]
    w = np.where(np.arange(K)[:, None, None] == K - 1, 0.0, w)
    out = ((1 - w) * fwd + w * bwd).reshape(K * D, 3)[:n]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return (down_sign * out).astype(np.float32)


#: Which frame the gyroscope is read in before any estimator runs (M19, lever 3).
#: "raw": as recorded. "truth": an ORACLE -- carried into the accelerometer's own
#: frame by the axis map fitted against ground truth, per recording. On most
#: drone flights the two sensors disagree (finding 21), and no IMU-only criterion
#: tried recovers the right map: three different ones agree with each other and
#: all get the x-axis sign wrong (`selfcal_likelihood_val.out`). So this prices
#: the fix and cannot be submitted: it needs `quat` and `pos`.
GYRO_FRAMES = ("raw", "truth")

#: Which gyroscope signal the estimator integrates (M19, lever 4). "denoised":
#: the rate corrected by the learned denoiser in `prep.gyro_denoise` (Brossard
#: et al.), trained on train against ground-truth rate. Independent of
#: `gyro_frame`; a second learned stage at inference, so see that module on
#: its legal standing before submitting it.
GYRO_SOURCES = ("raw", "denoised")


def _logmap(q: np.ndarray) -> np.ndarray:
    q = np.where(q[..., 3:4] < 0, -q, q)
    sn = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(sn, np.clip(q[..., 3:4], -1.0, 1.0))
    return q[..., :3] * np.where(sn > 1e-12, ang / np.maximum(sn, 1e-300), 2.0)


def _signed_perms() -> list[np.ndarray]:
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            P[range(3), perm] = signs
            out.append(P)
    return out


def _nearest_perm(R: np.ndarray) -> np.ndarray:
    perms = _signed_perms()
    return perms[int(np.argmax([np.sum(P * R) for P in perms]))]


def gyro_frame_from_truth(imu: np.ndarray, quat_xyzw: np.ndarray, pos: np.ndarray,
                          dt: float, cap: int = 60000) -> np.ndarray:
    """ORACLE: the signed axis map carrying this recording's gyro into its accelerometer's frame.

    Both sensors are fitted to ground truth separately -- the gyroscope to the
    body rate implied by `quat`, the accelerometer to the specific force implied
    by `pos` and `quat` -- each as the nearest signed axis permutation, and the
    map between them is the answer. The accelerometer fit is a proper rotation:
    allowing a mirror does not lower its residual (median ratio 1.000,
    `accel_frame_handedness_val.out`). Reads ground truth; diagnostic only.
    """
    m = min(len(imu), cap)
    h, k_w, k_a = 2, 10, 40
    x = np.asarray(imu[:m], np.float64)
    Q = np.asarray(quat_xyzw[:m], np.float64)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    def smooth(a, k):
        c = np.cumsum(np.vstack((np.zeros((1, a.shape[1])), a)), axis=0)
        return (c[k:] - c[:-k]) / k

    def fit(a, b, proper):
        u, _, vt = np.linalg.svd(b.T @ a)
        d = np.sign(np.linalg.det(u @ vt)) if proper else 1.0
        return _nearest_perm(u @ np.diag([1.0, 1.0, d]) @ vt)

    w_gt = smooth(_logmap(_qmul(_conj(Q[:-2 * h]), Q[2 * h:])) / (2 * h * dt), k_w)
    w_imu = smooth(x[:, 3:6], k_w)[h:h + len(w_gt)]
    p = np.asarray(pos[:m], np.float64)
    acc_w = (p[2 * h:] - 2 * p[h:-h] + p[:-2 * h]) / (h * dt) ** 2
    f_gt = smooth(_rotate(_conj(Q[h:m - h]), acc_w + np.array([0.0, 0.0, G])), k_a)
    f_imu = smooth(x[:, 0:3], k_a)[h:h + len(f_gt)]
    Ra = fit(f_imu, f_gt, proper=True)
    Og = fit(w_imu, w_gt, proper=False)
    return Ra.T @ Og


def estimator_spec(gravity_cfg: dict | None) -> dict:
    """`prep.gravity` -> the estimator's name and every parameter it uses.

    Only the estimator's own parameters are kept, so the cache key and the
    run log say exactly what produced the channels and nothing else.
    """
    cfg = dict(gravity_cfg or {})
    est = cfg.get("estimator", "cf")
    if est not in ESTIMATORS:
        raise KeyError(f"unknown prep.gravity.estimator {est!r}; have {list(ESTIMATORS)}")
    frame = cfg.get("gyro_frame", "raw")
    if frame not in GYRO_FRAMES:
        raise KeyError(f"unknown prep.gravity.gyro_frame {frame!r}; have {list(GYRO_FRAMES)}")
    gyro = cfg.get("gyro", "raw")
    if gyro not in GYRO_SOURCES:
        raise KeyError(f"unknown prep.gravity.gyro {gyro!r}; have {list(GYRO_SOURCES)}")
    # Stated only when not the default, so every earlier spec is unchanged.
    extra = {**({} if frame == "raw" else {"gyro_frame": frame}),
             **({} if gyro == "raw" else {"gyro": gyro})}
    if est == "box":
        return {"estimator": "box",
                "box_seconds": float(cfg.get("box_seconds", DEFAULT_BOX_SECONDS)), **extra}
    if est == "rts":
        return {"estimator": "rts", **{k: (int(cfg.get(k, v)) if k in ("block", "adaptive")
                                           else float(cfg.get(k, v)))
                                       for k, v in DEFAULT_RTS.items()}, **extra}
    return {"estimator": est, "alpha": float(cfg.get("alpha", DEFAULT_ALPHA)),
            "tolerance": float(cfg.get("tolerance", DEFAULT_TOLERANCE)), **extra}


def estimator_key(gravity_cfg: dict | None) -> str:
    """Cache-file tag for an estimator; empty for the default causal filter.

    The empty tag is what keeps the default cache at its old name,
    `gravity_body.npy`, so every earlier run and notebook still finds it.
    """
    s = estimator_spec(gravity_cfg)
    frame = ("+denoised" if s.get("gyro") == "denoised" else "") + \
            ("+truthframe" if s.get("gyro_frame") == "truth" else "")
    if s["estimator"] == "box":
        return f"box{s['box_seconds']:g}" + frame
    if s["estimator"] == "rts":
        extra = "".join(f"-{k}{s[k]:g}" for k, v in DEFAULT_RTS.items() if s[k] != v)
        return "rts" + extra + frame
    tag = "" if s["estimator"] == "cf" else s["estimator"]
    if (s["alpha"], s["tolerance"]) != (DEFAULT_ALPHA, DEFAULT_TOLERANCE):
        tag += f"-a{s['alpha']:g}-t{s['tolerance']:g}"
    return tag.lstrip("-") + frame


def estimate_gravity(imu: np.ndarray, dt: float, gravity_cfg: dict | None = None,
                     gyro_map: np.ndarray | None = None) -> np.ndarray:
    """One trajectory's `(N, 6)` IMU -> `(N, 3)` down-directions, by the named estimator.

    `gyro_map`, when given, is applied to the gyroscope first -- the oracle axis
    map of `gyro_frame_from_truth`. It moves only what the estimator reads; the
    network's own six raw channels are never touched.
    """
    s = estimator_spec(gravity_cfg)
    x = np.asarray(imu, np.float64)[:, :6]
    if gyro_map is not None:
        x = x.copy()
        x[:, 3:6] = x[:, 3:6] @ np.asarray(gyro_map, np.float64).T
    if s["estimator"] == "box":
        return gravity_box(x, dt, s["box_seconds"])
    if s["estimator"] == "rts":
        return gravity_rts(x, dt, **{k: s[k] for k in DEFAULT_RTS})
    if s["estimator"] == "cf_fb":
        return gravity_forward_backward(x, dt, s["alpha"], s["tolerance"])
    return gravity_direction(x, dt, s["alpha"], s["tolerance"])


# --------------------------------------------------------------------- ground truth

def gravity_from_quaternion(quat_xyzw: np.ndarray,
                            down_sign: float = DOWN_SIGN) -> np.ndarray:
    """The true down direction in body coordinates, from ground-truth attitude.

    Only train and val carry `quat`, so this is a yardstick for `verify()` and
    the reference `gravity_from_quaternion_torch` below is tested against --
    **never a model input.**

    The stored quaternion rotates body coordinates into world coordinates
    (CLAUDE.md finding 4). Applying its inverse to the world's vertical axis
    therefore expresses that axis in body coordinates. `down_sign` selects which
    end of the axis we call "down", so this and `gravity_direction` agree on the
    same convention by construction.

    Computed in closed form -- the third row of R(q), exactly what
    `gravity_from_quaternion_torch` below computes -- rather than through
    scipy's `Rotation`, which gave the same numbers but does not load in this
    project's venv on macOS 27 (a user-site scipy binary the new dyld rejects),
    and so broke `verify()` and the truth oracle on the laptop. Quaternions are
    normalised first, as `Rotation.from_quat` did.
    """
    q = np.asarray(quat_xyzw, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    body_axis = np.stack((2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
                          1.0 - 2.0 * (x * x + y * y)), axis=-1)
    return (body_axis * down_sign).astype(np.float32)


def degrade_gravity(g: np.ndarray, median_deg: float, tau_s: float, dt: float,
                    seed_key: str) -> np.ndarray:
    """Ground-truth down plus a slowly drifting tilt error whose median is `median_deg`.

    For M20's dose-response: M16 fed PERFECT gravity and moved drone by 0.065
    m/s. A real estimator is never perfect, and its error is not white noise --
    it drifts over seconds, the way a filter's does when the platform
    manoeuvres. So the error here is a smooth random process: three
    independent AR(1) signals with correlation time `tau_s`, sampled every 0.1 s
    and interpolated to the frame rate, projected onto the plane perpendicular
    to gravity (only tilt is perturbed, never length), and scaled so the tilt
    angle is Rayleigh-distributed with median `median_deg`.

    Deterministic per trajectory: the draw is seeded from `seed_key` (the
    trajectory id) through a stable hash, so every chunk, every epoch and every
    session sees the same error at the same frame.
    """
    import zlib

    g = np.asarray(g, np.float64)
    n = len(g)
    if n == 0 or median_deg <= 0:
        return g.astype(np.float32)
    rng = np.random.default_rng(np.random.SeedSequence(
        [20260920, zlib.crc32(str(seed_key).encode())]))
    step = max(1, int(round(0.1 / dt)))
    m = n // step + 2
    rho = float(np.exp(-0.1 / tau_s))
    knots = np.empty((m, 3))
    knots[0] = rng.normal(size=3)
    innov = rng.normal(size=(m, 3)) * np.sqrt(1.0 - rho * rho)
    for k in range(1, m):
        knots[k] = rho * knots[k - 1] + innov[k]
    pos = np.arange(n) / step
    k0 = pos.astype(np.int64)
    fr = (pos - k0)[:, None]
    e = (1.0 - fr) * knots[k0] + fr * knots[k0 + 1]
    t = e - np.sum(e * g, axis=1, keepdims=True) * g        # tangent to the sphere at g
    mag = np.linalg.norm(t, axis=1, keepdims=True)
    # |t| is Rayleigh with unit scale, median sqrt(2 ln 2); rescale to the target.
    phi = mag * np.radians(median_deg) / np.sqrt(2.0 * np.log(2.0))
    tu = t / np.maximum(mag, 1e-12)
    out = np.cos(phi) * g + np.sin(phi) * tu
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def gravity_from_quaternion_torch(q_xyzw, down_sign: float = DOWN_SIGN):
    """`gravity_from_quaternion`, in closed form: batched, differentiable, no scipy.

    `(..., 4)` quaternions `[x, y, z, w]` -> `(..., 3)` down-directions in body
    coordinates. This is the training target of the auxiliary gravity head
    (`models.heads.GravityHead`, M15), computed on the fly from the `q_gt` the
    batch already carries, so it needs no cache and no new plumbing.

    The quaternion rotates body into world, so its rotation matrix R does too,
    and the world's vertical axis seen from the body is `R^T e_z` -- the third
    *row* of R:

        g = down_sign * [ 2(xz - wy),  2(yz + wx),  1 - 2(x^2 + y^2) ]

    `tests/test_m15.py` checks this against scipy and against the function
    above to floating point, because a swapped index or sign here would be
    silent: the head would learn a perfectly good answer to the wrong question.

    **Zero rows are the caller's to exclude, not this function's.** `q_gt` is
    all zeros on test rows and on padding, so the caller masks by the
    quaternion's own norm first. Here the input is normalised with a clamped
    divisor, which leaves a real (near-unit) quaternion unchanged and turns a
    zero row into a harmless finite vector rather than a NaN.
    """
    import torch

    q = q_xyzw / q_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    g = torch.stack((2.0 * (x * z - w * y),
                     2.0 * (y * z + w * x),
                     1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return g * down_sign


# --------------------------------------------------------------------------- cache

def orientation_path(split: str, root: Path | None = None, key: str = "") -> Path:
    """The cache file for one split and one estimator.

    `key` is `estimator_key(prep.gravity)`: empty for the default causal
    filter, whose file keeps its original name, `gravity_body.npy`.
    """
    name = "gravity_body.npy" if not key else f"gravity_body@{key}.npy"
    return (root or CACHE_V) / split / name


def orientation_available(split: str, root: Path | None = None, key: str = "") -> bool:
    return orientation_path(split, root, key).exists()


def load_orientation(split: str, root: Path | None = None, key: str = "") -> np.ndarray:
    """`(F, 3)` float16 memmap of per-frame down-direction, frame-major like the IMU."""
    path = orientation_path(split, root, key)
    if not path.exists():
        flag = f" --estimator ...  (the estimator whose key is {key!r})" if key else ""
        raise FileNotFoundError(
            f"no gravity cache for split {split!r} at {path}. Build it with:\n"
            f"    mySolution/.venv/bin/python -m src.prep.orientation{flag}")
    return np.load(path, mmap_mode="r")


def build_split(split: str, root: Path | None = None, force: bool = False,
                alpha: float = DEFAULT_ALPHA, tolerance: float = DEFAULT_TOLERANCE,
                verbose: bool = True, gravity_cfg: dict | None = None) -> dict:
    """Run the estimator over every trajectory in one split and cache the result.

    With no `gravity_cfg` this is exactly the original build: the causal filter
    at `alpha`/`tolerance`, written to `gravity_body.npy`. With one, the estimator
    `prep.gravity` names is run instead and written to its own keyed file, so a
    two-sided arm can never overwrite the cache every earlier run reads.
    """
    spec = (estimator_spec(gravity_cfg) if gravity_cfg is not None else
            {"estimator": "cf", "alpha": float(alpha), "tolerance": float(tolerance)})
    key = estimator_key(spec)
    path = orientation_path(split, root, key)
    if path.exists() and not force:
        if verbose:
            print(f"[{split}] gravity cache {path.name} exists, skipping "
                  f"(use --force to rebuild)")
        return {"split": split, "skipped": True, "key": key}

    cache = load_cache(split)
    oracle = spec.get("gyro_frame") == "truth"
    if oracle and cache.quat_gt is None:
        raise ValueError(
            f"gyro_frame='truth' fits each recording's gyro frame against ground truth, "
            f"and split {split!r} carries none. An oracle cache cannot exist for test.")
    dt = 1.0 / float(cache.meta.get("sampling_rate_hz", 200.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(cache.imu.shape[0], 3))

    denoiser = None
    if spec.get("gyro") == "denoised":
        from .gyro_denoise import denoise, load_denoiser
        denoiser = load_denoiser(root)

    t0 = time.perf_counter()
    for t in range(cache.n_traj):
        row = cache.trajectories.iloc[t]
        lo = int(row.frame_offset)
        hi = lo + int(row.n_frames_kept)
        frames = np.asarray(cache.imu[lo:hi], np.float64)
        if denoiser is not None:
            frames = denoise(frames, denoiser)
        gmap = None
        # The oracle corrects DRONE flights only. The defect is measured on drone
        # alone (car/dog/human gyro-vs-accelerometer 0.6-3 deg), and car and dog
        # positions are too noisy to differentiate twice (a 5 m/s^2 fit residual),
        # so fitting their accelerometer frame from them lands on wrong axes: it
        # moved car 0.27 -> 0.41 deg before this restriction. An oracle may read
        # the platform label; nothing on a submission path does.
        if oracle and str(row.platform) == "drone":
            from ..paths import traj_npz
            raw = np.load(traj_npz(split, str(row.platform), str(row.traj_id)))
            n_kept = int(row.n_frames_kept)
            gmap = gyro_frame_from_truth(frames, raw["quat"][:n_kept], raw["pos"][:n_kept], dt)
        out[lo:hi] = estimate_gravity(frames, dt, spec, gyro_map=gmap).astype(np.float16)
        if verbose and (t + 1) % 100 == 0:
            print(f"  [{split}] {t + 1}/{cache.n_traj} trajectories")
    out.flush()
    del out

    meta = {
        "split": split,
        "filter": ("complementary_direction_only" if spec["estimator"] == "cf"
                   else spec["estimator"]),
        **spec, "key": key, "down_sign": DOWN_SIGN,
        "dt": dt, "n_frames": int(cache.imu.shape[0]),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "built_by": env_record(),
    }
    write_json(path.with_suffix(".json"), meta)
    if verbose:
        print(f"[{split}] {meta['n_frames']:,} frames in {meta['build_seconds']}s "
              f"-> {path.name}")
    return meta


def build_all(splits=SPLITS, root: Path | None = None, force: bool = False,
              alpha: float = DEFAULT_ALPHA, tolerance: float = DEFAULT_TOLERANCE,
              verbose: bool = True, gravity_cfg: dict | None = None) -> dict[str, dict]:
    return {s: build_split(s, root, force, alpha, tolerance, verbose, gravity_cfg)
            for s in splits}


# ------------------------------------------------------------------- verification

def verify(split: str = "val", per_platform_n: int = 4, seed: int = 0,
           root: Path | None = None, key: str = "") -> dict:
    """Compare the filter's answer against ground truth, per platform.

    GravityDirectionPlan.md calls this the first task, and for a good reason: a
    flipped sign or a swapped axis produces a perfectly plausible-looking cache
    that poisons every run afterwards. The check reports the angle between the
    filtered direction and the true one, in degrees, and it also reports what the
    *opposite* sign convention would have scored -- if the wrong one is winning,
    the numbers say so instead of the model quietly suffering.

    Trajectories are drawn **evenly across the four platforms**, not uniformly.
    Val is 60% drone by trajectory count and drone is the hardest platform for
    any orientation filter, so a uniform draw would report drone's number with
    three other platforms' names attached to it.
    """
    cache = load_cache(split)
    if cache.quat_gt is None:
        raise ValueError(f"split {split!r} has no ground-truth attitude to check against")
    grav = load_orientation(split, root, key)

    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for platform in PLATFORMS:
        rows = np.flatnonzero((cache.trajectories.platform == platform).to_numpy())
        if len(rows):
            picks += [int(i) for i in rng.choice(
                rows, min(per_platform_n, len(rows)), replace=False)]
    per_platform: dict[str, list[float]] = {}
    flipped: list[float] = []

    for t in picks:
        row = cache.trajectories.iloc[int(t)]
        lo = int(row.frame_offset)
        hi = lo + int(row.n_frames_kept)
        truth = gravity_from_quaternion(np.asarray(cache.quat_gt[lo:hi], np.float64))
        ours = np.asarray(grav[lo:hi], np.float64)
        ours /= np.maximum(np.linalg.norm(ours, axis=1, keepdims=True), 1e-9)

        cos = np.clip((ours * truth).sum(axis=1), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        per_platform.setdefault(str(row.platform), []).append(float(np.median(angle)))
        flipped.append(float(np.median(np.degrees(np.arccos(np.clip(-cos, -1, 1))))))

    summary = {p: round(float(np.mean(v)), 3) for p, v in sorted(per_platform.items())}
    # Macro-average over platforms, matching how the competition metric reads.
    macro = float(np.mean(list(summary.values())))
    return {
        "split": split,
        "n_trajectories": len(picks),
        "median_angle_deg_per_platform": summary,
        "median_angle_deg": round(macro, 3),
        "median_angle_deg_if_sign_flipped": round(float(np.mean(flipped)), 3),
        "sign_convention_is_correct": macro < float(np.mean(flipped)),
    }


# --------------------------------------------------------------------------- entrypoint

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=SPLITS, action="append", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--verify", action="store_true",
                    help="check the cached directions against ground-truth attitude")
    ap.add_argument("--estimator", choices=ESTIMATORS, default=None,
                    help="build a two-sided estimator's own cache instead of the default filter's")
    ap.add_argument("--box-seconds", type=float, default=DEFAULT_BOX_SECONDS)
    a = ap.parse_args()

    gcfg = None
    if a.estimator is not None:
        gcfg = {"estimator": a.estimator}
        if a.estimator == "box":
            gcfg["box_seconds"] = a.box_seconds
        elif a.estimator in ("cf", "cf_fb"):
            gcfg.update({"alpha": a.alpha, "tolerance": a.tolerance})
    key = estimator_key(gcfg) if gcfg is not None else estimator_key(
        {"alpha": a.alpha, "tolerance": a.tolerance})
    if not a.verify:
        build_all(a.split or SPLITS, force=a.force,
                  alpha=a.alpha, tolerance=a.tolerance, gravity_cfg=gcfg)
    for split in (a.split or ["val"]):
        if split in ("train", "val"):
            report = verify(split, key=key)
            print(f"\n[{split}] median angle to truth: "
                  f"{report['median_angle_deg']:.2f} deg   "
                  f"(wrong sign would give {report['median_angle_deg_if_sign_flipped']:.2f})")
            for p, v in report["median_angle_deg_per_platform"].items():
                print(f"    {p:<7} {v:6.2f} deg")


if __name__ == "__main__":
    main()
