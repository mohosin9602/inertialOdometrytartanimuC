"""Augmentation: time scaling -- the same path, travelled faster. `.claude/augmentation.md` §17.

**The one idea.** Play a recording back `s` times faster. The platform follows
exactly the same path through exactly the same attitudes, only sooner, so every
quantity the sensor measures changes by a known power of `s`:

    velocity (the target)     x s      the same distance in 1/s of the time
    gyroscope                 x s      the same turns, taken faster
    linear acceleration       x s^2    s times more speed to change, in 1/s the time
    gravity                   unchanged -- driving faster does not make Earth pull harder

Nothing is approximated beyond resampling: the sped-up recording is what the same
sensor would read if carried along the same path `s` times faster, and the new
label is exact. `tests/test_timescale.py` proves it against ground truth.

**Why the accelerometer must be split first.** It reads gravity plus motion as one
vector. Scaling the whole reading by `s^2` would turn 9.81 m/s^2 of gravity into
19.2 at `s = 1.4`, which no sensor on Earth reads; scaling it by `s` gets the motion
wrong. So gravity is taken out (box5's two-sided estimate, the best IMU-only one),
only the motion is scaled, and gravity goes back in unchanged.

**Vibration is not motion (`vibkeep`).** No platform's vibration frequency follows
its speed (elasticity <= +0.16, measured), so a naive speed-up would move a drone's
29 Hz motor tone to 41 Hz -- a fingerprint the embodiment branch reads, and one no
real drone has. Only the part below `fc_hz` (8 Hz) is time-scaled. The part above
is copied at its own speed, one window at a time, from the stretch of recording the
window's motion came from -- so the vibration still belongs to that moment of the
flight (motors idling on the ground, loud in the air).

**Only drone, by default -- and only because it was measured.** Drone is the one
platform whose test split moves harder than train (linear acceleration 1.87x,
vibration 1.15x), and the one whose real speed-ups look like a time-lapse
(acceleration elasticity +1.58, turning +0.91). The robot dog steps at a fixed
rhythm whatever its speed, so a sped-up dog is a dog that does not exist.

**The gravity channels are re-estimated, never resampled.** `ChunkDataset` runs the
run's own estimator on the sped-up IMU. On faster motion the filter really is
worse (drone 9-10 deg against a resampled channel's 7.3 deg), and handing the
network a channel more accurate than it can be at test time teaches over-trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..paths import FS, PLATFORMS, WIN, traj_npz
from ..prep.orientation import gravity_box

G0 = 9.81

#: Named, so a run record says what it trained on in one word. The magnitudes are
#: `.claude/augmentation.md` §17.4's: sqrt(1.87) = 1.37 moves train's median drone
#: acceleration onto test's, and realism degrades past ~1.4 (35% of spans outside
#: the real 5-95% band at s = 1.5, against 11% for real data).
RECIPES: dict[str, dict] = {
    "timescale_drone": {
        "s_range": {"car": [1.0, 1.0], "dog": [1.0, 1.0], "drone": [1.0, 1.4],
                    "human": [1.0, 1.0]},
        "p": {"car": 0.0, "dog": 0.0, "drone": 0.5, "human": 0.0},
        "fc_hz": 8.0,
        "split_box_seconds": 5.0,
    },
}


# ------------------------------------------------------------------ signal helpers

def band(x: np.ndarray, lo: float, hi: float, fs: float = FS) -> np.ndarray:
    """Rows of `x` (n, c) band-passed to [lo, hi) Hz; `lo = 0` keeps DC.

    A brick wall in the FFT, over a copy reflected at both ends so the chunk's two
    edges are continuous and do not ring into each other. `band(x, 0, f) +
    band(x, f, inf)` is `x` exactly, which is what lets the two halves be treated
    differently and still add back up.
    """
    x = np.asarray(x, np.float64)
    n = len(x)
    pad = max(0, min(n - 2, 512))
    xp = (np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]], axis=0) if pad else x)
    m = len(xp)
    f = np.fft.rfftfreq(m, 1.0 / fs)
    keep = (f < hi) & (f >= lo) if lo > 0 else (f < hi)
    y = np.fft.irfft(np.fft.rfft(xp, axis=0) * keep[:, None], m, axis=0)
    return y[pad:pad + n]


def interp_rows(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rows of `x` at fractional frame times `t`, linearly interpolated."""
    t = np.asarray(t, np.float64)
    i0 = np.clip(np.floor(t).astype(np.int64), 0, len(x) - 2)
    w = (t - i0)[:, None]
    return x[i0] * (1.0 - w) + x[i0 + 1] * w


def nlerp(q_xyzw: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Quaternions at fractional frame times: normalised linear interpolation.

    Neighbouring frames are 5 ms apart, where nlerp and slerp agree to rounding.
    The second quaternion is sign-aligned first, because `q` and `-q` are the same
    rotation and averaging across that flip would pass through zero.
    """
    q = np.asarray(q_xyzw, np.float64)
    t = np.asarray(t, np.float64)
    i0 = np.clip(np.floor(t).astype(np.int64), 0, len(q) - 2)
    w = (t - i0)[:, None]
    a, b = q[i0], q[i0 + 1]
    b = np.where((a * b).sum(1, keepdims=True) < 0, -b, b)
    out = a * (1.0 - w) + b * w
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def out_frames(n_in: int, s: float) -> int:
    """How many new frames `n_in` original frames yield at speed `s`."""
    return int(np.floor((n_in - 1) / s)) + 1


def vibration_index(n_in: int, n_out: int, s: float) -> np.ndarray:
    """Where each new frame's vibration is copied from, at native speed.

    New window `j` takes the 200 consecutive original frames centred on the
    original moment its own middle maps to -- so the vibration belongs to the same
    point of the path as the motion, while keeping its own frequency.
    """
    j = np.arange(n_out // WIN)
    start = np.round(s * (j * WIN + WIN / 2)).astype(np.int64) - WIN // 2
    start = np.clip(start, 0, max(n_in - WIN, 0))
    return (start[:, None] + np.arange(WIN)[None, :]).reshape(-1)


def time_scale(acc: np.ndarray, gyr: np.ndarray, down: np.ndarray, s: float,
               n_out: int, fc_hz: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The transform. Returns the new accelerometer, gyroscope and down-direction.

    `acc`, `gyr`: (n, 3) in the sensor frame. `down`: (n, 3) unit vectors along the
    at-rest accelerometer reading -- used ONLY to split gravity from motion, never
    shown to the model. `n_out` must be a whole number of windows and no more than
    `out_frames(n, s)`.
    """
    n = len(acc)
    if n_out > out_frames(n, s) or n_out % WIN:
        raise ValueError(f"{n_out} new frames from {n} at s={s:.3f}: at most "
                         f"{out_frames(n, s)}, in whole windows of {WIN}")
    t = s * np.arange(n_out)
    lin = np.asarray(acc, np.float64) - G0 * np.asarray(down, np.float64)
    gyr = np.asarray(gyr, np.float64)
    lin_lo, lin_hi = band(lin, 0.0, fc_hz), band(lin, fc_hz, np.inf)
    gyr_lo, gyr_hi = band(gyr, 0.0, fc_hz), band(gyr, fc_hz, np.inf)
    vib = vibration_index(n, n_out, s)
    d = interp_rows(np.asarray(down, np.float64), t)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    acc_new = G0 * d + s * s * interp_rows(lin_lo, t) + lin_hi[vib]
    gyr_new = s * interp_rows(gyr_lo, t) + gyr_hi[vib]
    return acc_new, gyr_new, d


def scaled_truth(truth: dict, f0: int, n_in: int, s: float, n_out: int) -> dict:
    """The labels of the sped-up recording, in the pinned conventions (finding 3).

    `v_gt` is the per-window MEAN of the new body velocity; `q_gt` the attitude at
    each new window's middle; `p_gt` the position at its end. Positions are the
    same path, sampled at the new instants.
    """
    sl = slice(f0, f0 + n_in)
    k = n_out // WIN
    vel = s * interp_rows(truth["vel"][sl], s * np.arange(n_out))
    mid = s * (np.arange(k) * WIN + WIN // 2)
    end = np.minimum(s * (np.arange(k) + 1) * WIN, n_in - 1)
    return {
        "v_gt": vel.reshape(k, WIN, 3).mean(axis=1).astype(np.float32),
        "q_gt": nlerp(truth["quat"][sl], mid).astype(np.float32),
        "p_gt": (interp_rows(truth["pos"][sl], end) + truth["pos0"]).astype(np.float32),
    }


def trajectory_truth(cache, t: int, box_seconds: float = 5.0) -> dict:
    """What the transform needs from one trajectory, loaded once and kept.

    Per-frame body velocity, position and attitude come from the raw NPZ -- the
    Tier 0 cache holds only window means, and a sped-up window straddles two
    original windows. The file is matched to the cache row the same way the cache
    builder checks itself: its window means must BE the cached targets. The
    gravity split is box5 over the whole trajectory's cached frames.
    """
    row = cache.trajectories.iloc[t]
    lo, n, nw = int(row.frame_offset), int(row.n_frames_kept), int(row.n_windows)
    z = np.load(traj_npz(cache.split, str(row.platform), str(row.traj_id)))
    vel = np.asarray(z["vel_body"][:n], np.float64)
    own = vel.reshape(nw, WIN, 3).mean(axis=1)
    gap = float(np.abs(own - np.asarray(cache.v_gt[cache.windows_slice(t, 0, nw)])).max())
    if gap > 1e-3:
        raise ValueError(f"{row.traj_id}: raw vel_body window means differ from the cached "
                         f"targets by {gap:.2e} -- not the trajectory this cache row holds")
    pos = np.asarray(z["pos"][:n], np.float64)
    quat = np.asarray(z["quat"][:n], np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    frames = np.asarray(cache.imu[lo:lo + n], np.float64)
    return {"vel": vel.astype(np.float32), "pos0": pos[0].copy(),
            # float32 relative to the start keeps millimetres over a kilometre
            "pos": (pos - pos[0]).astype(np.float32), "quat": quat.astype(np.float32),
            "down": gravity_box(frames, 1.0 / FS, box_seconds).astype(np.float32)}


# ------------------------------------------------------------------------ the draw

@dataclass
class TimeScaler:
    """Draws one speed factor per chunk, reproducibly -- or None to leave it alone.

    Like `augment.Augmenter`, the draw depends on `(seed, epoch, chunk index)` and
    nothing else, so it is independent of batch order; a separate stream (the
    fourth seed word) keeps it independent of the re-mounting draw too. `s` is
    log-uniform, so speeding up by 1.2 and by 1/1.2 would be equally likely were
    both allowed.
    """
    s_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    fc_hz: float = 8.0
    split_box_seconds: float = 5.0
    seed: int = 42
    epoch: int = 0
    # M23: the probability, per platform, that a chunk's window grid is shifted by
    # a random 1-199 frames. Real data re-cut, not a transform: the same frames and
    # the same per-frame truth, only the window boundaries move. Empty = off, and
    # then nothing below draws from its stream, so every M22 run is unchanged.
    phase_p: dict[str, float] = field(default_factory=dict)

    STREAM = 0x7153        # "ts": keeps this generator apart from the re-mounting one
    PHASE_STREAM = 0x9A5E  # a third, independent stream for the window phase

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def enabled(self) -> bool:
        return (any(self.p.get(q, 0.0) > 0 and self._range(q)[0] != self._range(q)[1]
                    for q in PLATFORMS)
                or any(self.phase_p.get(q, 0.0) > 0 for q in PLATFORMS))

    def draw_phase(self, index: int, platform: str) -> int:
        """Frames to shift chunk `index`'s window grid by this epoch: 0, or 1..199."""
        p = float(self.phase_p.get(platform, 0.0))
        if p <= 0.0:
            return 0
        rng = np.random.default_rng([int(self.seed), int(self.epoch), int(index),
                                     self.PHASE_STREAM])
        if rng.random() >= p:
            return 0
        return int(rng.integers(1, WIN))

    def _range(self, platform: str) -> tuple[float, float]:
        lo, hi = self.s_range.get(platform, (1.0, 1.0))
        return float(lo), float(hi)

    def draw(self, index: int, platform: str) -> float | None:
        p = float(self.p.get(platform, 0.0))
        lo, hi = self._range(platform)
        if p <= 0.0 or lo == hi:
            return None
        rng = np.random.default_rng([int(self.seed), int(self.epoch), int(index), self.STREAM])
        if rng.random() >= p:
            return None
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def build_timescaler(cfg) -> TimeScaler | None:
    """The time scaler a config asks for, or None. Absent keys mean off."""
    aug = cfg.get("augment", {}) or {}
    if "timescale" not in list(aug.get("ops", []) or []):
        return None
    spec = dict(aug.get("timescale", {}) or {})
    recipe = spec.pop("recipe", None)
    if recipe is not None:
        if recipe not in RECIPES:
            raise KeyError(f"unknown timescale recipe {recipe!r}; have {sorted(RECIPES)}")
        spec = {**RECIPES[recipe], **spec}
    s_range = {q: tuple(float(v) for v in spec.get("s_range", {}).get(q, (1.0, 1.0)))
               for q in PLATFORMS}
    for q, (lo, hi) in s_range.items():
        if not 0.5 <= lo <= hi <= 2.0:
            raise ValueError(f"timescale s_range for {q} is {lo}..{hi}; want 0.5 <= lo <= hi <= 2")
    phase_p = {q: float((spec.get("phase_p", {}) or {}).get(q, 0.0)) for q in PLATFORMS}
    for q, v in phase_p.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"timescale phase_p for {q} is {v}; want 0 <= p <= 1")
    ts = TimeScaler(s_range=s_range,
                    p={q: float(spec.get("p", {}).get(q, 0.0)) for q in PLATFORMS},
                    fc_hz=float(spec.get("fc_hz", 8.0)),
                    split_box_seconds=float(spec.get("split_box_seconds", 5.0)),
                    seed=int(cfg["run"]["seed_augment"]),
                    phase_p=phase_p)
    return ts if ts.enabled else None


# ----------------------------------------------------------------------- the gate

def _rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64).T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    -1).reshape(-1, 3, 3)


def _d2(p: np.ndarray, win: int = 41, order: int = 3) -> np.ndarray:
    """Savitzky-Golay second derivative per axis; the ends are left at zero."""
    h = win // 2
    k = np.linalg.pinv(np.vander(np.arange(-h, h + 1), order + 1, increasing=True))[2]
    k = k * 2 * FS ** 2
    out = np.zeros_like(p, dtype=np.float64)
    for j in range(p.shape[1]):
        out[h:-h, j] = np.convolve(p[:, j], k[::-1], mode="valid")
    return out


def synthesize_imu(quat: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A noise-free IMU built FROM ground truth: specific force, body rate, down."""
    R = _rotmat(quat)
    acc = np.einsum("nji,nj->ni", R, _d2(np.asarray(pos, np.float64)) + np.array([0, 0, G0]))
    dR = np.einsum("nji,njk->nik", R[:-4], R[4:])
    gyr = np.zeros_like(acc)
    gyr[2:-2] = np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0],
                          dR[:, 1, 0] - dR[:, 0, 1]], 1) / 2 / (4 / FS)
    down = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
    return acc, gyr, down


def gate(cache, platform: str = "drone", n_traj: int = 6, scales=(1.25, 1.4),
         seed: int = 0) -> dict:
    """The correctness gate, on real ground truth. Returns medians over recordings.

    exact     the transform on a noise-free IMU synthesised from ground truth,
              against the same synthesis of the sped-up motion, below 2 Hz. A
              correct transform reads 0.03-0.06 m/s^2 and ~0.001 rad/s.
    bug_*     the same with a planted bug -- linear acceleration scaled by s instead
              of s^2, and the gyroscope left unscaled. They must read far larger,
              or the gate proves nothing.
    velocity  R(q_new) v_new against the central difference of the new positions,
              the CLAUDE.md gate, on the recorded (not synthetic) truth (m/s).
    """
    rng = np.random.default_rng(seed)
    rows = cache.trajectories
    ids = [t for t in range(cache.n_traj) if str(rows.platform.iloc[t]) == platform]
    out = {k: [] for k in ("acc", "gyr", "bug_linear_s", "bug_gyro_unscaled", "velocity")}
    keep = slice(60, -60)

    def rms(a, b):
        return float(np.sqrt(np.mean(np.sum((band(a, 0, 2.0)[keep] - band(b, 0, 2.0)[keep]) ** 2, 1))))

    for t in rng.choice(ids, min(n_traj, len(ids)), replace=False):
        truth = trajectory_truth(cache, int(t))
        n = min(len(truth["vel"]), 90 * WIN)
        q = truth["quat"][:n].astype(np.float64)
        p = truth["pos"][:n].astype(np.float64)
        acc, gyr, down = synthesize_imu(q, p)
        for s in scales:
            m = out_frames(n, s) // WIN * WIN
            a1, g1, _ = time_scale(acc, gyr, down, s, m)
            tt = s * np.arange(m)
            q2, p2 = nlerp(q, tt), interp_rows(p, tt)
            a2, g2, _ = synthesize_imu(q2, p2)
            out["acc"].append(rms(a1, a2))
            out["gyr"].append(rms(g1, g2))
            lin = band(acc - G0 * down, 0.0, 8.0)
            d = interp_rows(down, tt)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            out["bug_linear_s"].append(rms(G0 * d + s * interp_rows(lin, tt), a2))
            out["bug_gyro_unscaled"].append(rms(interp_rows(band(gyr, 0.0, 8.0), tt), g2))
            v2 = s * interp_rows(truth["vel"][:n].astype(np.float64), tt)
            vw = np.einsum("nij,nj->ni", _rotmat(q2), v2)
            cd = (p2[2:] - p2[:-2]) / (2.0 / FS)
            out["velocity"].append(float(np.sqrt(np.mean(np.sum((vw[1:-1] - cd) ** 2, 1)))))
    return {k: float(np.median(v)) for k, v in out.items()}


#: What `gate` must read, used by the test suite and the notebook preflight alike.
GATE_LIMITS = {"acc": 0.15, "gyr": 0.01, "velocity": 0.02,
               "bug_linear_s": 0.25, "bug_gyro_unscaled": 0.05}


def check_gate(result: dict) -> list[str]:
    """Human-readable failures of a `gate` result; an empty list means it passed."""
    fails = []
    for k in ("acc", "gyr", "velocity"):
        if not result[k] < GATE_LIMITS[k]:
            fails.append(f"{k} {result[k]:.4f} >= {GATE_LIMITS[k]}")
    for k in ("bug_linear_s", "bug_gyro_unscaled"):
        if not result[k] > GATE_LIMITS[k]:
            fails.append(f"planted {k} reads only {result[k]:.4f} <= {GATE_LIMITS[k]}: "
                         f"the gate is not sharp")
    return fails
