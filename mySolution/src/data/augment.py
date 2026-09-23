"""Augmentation: the re-mounting family. PipelinePlan.md §7, `.claude/augmentation.md`.

**The one idea.** Every op here is the same physical statement: *the same motion,
recorded by the same sensor bolted on in a different orientation.* Take a
constant 3x3 map `A`, apply it to every body-frame vector in the sample -- the
accelerometer, the gyroscope, the gravity channel, and the target velocity --
and what comes out is a recording that could really have happened. Nothing is
approximated and nothing is invented; the world-frame trajectory does not move
at all.

That matters because the alternative -- perturbing the input and leaving the
label alone -- teaches a mapping that is physically impossible. §7 of
PipelinePlan.md names that trap explicitly, and `tests/test_augment.py` is the
gate that proves we did not fall into it: the augmented body velocity, rotated
into the world by the augmented attitude, must still equal the world velocity
obtained by differencing the (unaugmented) ground-truth positions.

**Three ops, one map.**

| op | what it simulates | axis |
| --- | --- | --- |
| `yaw` | the sensor rotated about the vertical | the estimated gravity direction |
| `tilt` | the sensor bolted on a few degrees off level | a random axis perpendicular to gravity |
| `mirror` | the whole scene reflected left/right | the vertical plane through the body's forward axis |

`yaw` is the one with a measured justification per platform. Drone's body-frame
velocity azimuth is very nearly uniform (resultant length 0.14-0.16, against
car's 0.62), so rotating a drone about its own vertical produces a sample drawn
from the same distribution -- free coverage, aimed exactly at the directional
train/val gap of CLAUDE.md finding 17. Car, dog and human have a real forward
prior, so they get a small jitter that simulates mounting tolerance rather than
a full circle that would erase the prior. The numbers are in
`.claude/augmentation.md`.

**Why this is free.** The three gravity channels come from a complementary
filter, and that filter is exactly equivariant: rotating its input by a constant
`A` rotates its output by the same `A`, measured at 5.9e-08 (a float32 rounding)
for proper rotations and 0.0e+00 for reflections. So the channels can be rotated
in place and the filter never runs again. Had it not held, every augmented chunk
would have cost a fresh filter pass -- about 25 s of CPU per training epoch.
`.claude/agentTests/2026-09-17_augmentation/check_filter_equivariance.py`.

**The pseudovector sign is the silent bug in the room.** Under a reflection the
accelerometer transforms as `M a` but the gyroscope as `-M w`, because angular
velocity is a pseudovector. Forgetting that sign does not crash anything: it
moves the filter's answer by 0.08 to 1.19 (against unit-length vectors), and the
model simply learns a wrong world. The same check pins it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..paths import PLATFORMS

#: Recipes are named so a run record says what it trained on in one word.
#: `.claude/augmentation.md` holds the argument for each number.
RECIPES: dict[str, dict] = {
    # The default. Rotation only: exact, free, and every magnitude is justified
    # by the per-platform velocity-azimuth measurement in augmentation.md §2.
    "remount_v1": {
        "yaw_deg": {"car": 10.0, "dog": 15.0, "drone": 180.0, "human": 10.0},
        "tilt_deg": 5.0,
        "mirror_p": {"car": 0.0, "dog": 0.0, "drone": 0.0, "human": 0.0},
    },
    # Drone only: the narrowest arm that still targets finding 17. Useful if the
    # default turns out to cost the three prior-carrying platforms.
    "remount_drone": {
        "yaw_deg": {"car": 0.0, "dog": 0.0, "drone": 180.0, "human": 0.0},
        "tilt_deg": 0.0,
        "mirror_p": {"car": 0.0, "dog": 0.0, "drone": 0.0, "human": 0.0},
    },
    # Everything the family can do, mirrors included. Mirrors are off by default
    # because human's azimuth distribution is measurably NOT mirror-symmetric
    # (total variation 0.57 against its own reflection, where drone is 0.17).
    "remount_mirror": {
        "yaw_deg": {"car": 10.0, "dog": 15.0, "drone": 180.0, "human": 10.0},
        "tilt_deg": 5.0,
        "mirror_p": {"car": 0.5, "dog": 0.5, "drone": 0.5, "human": 0.0},
    },
}

#: Reflection of the WORLD frame that pairs with a body-frame mirror. Any
#: reflection across a vertical plane works -- the augmented recording is valid
#: for all of them -- so the simplest is chosen. It must fix the world vertical,
#: or the augmented trajectory would fall up.
WORLD_MIRROR = np.diag([1.0, -1.0, 1.0])


# --------------------------------------------------------------- small rotation maths

def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by `angle` radians about `axis`, as a 3x3 matrix."""
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    """`(N,4)` [x,y,z,w] -> `(N,3,3)` rotation matrices, body -> world."""
    q = np.asarray(q_xyzw, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """`(N,3,3)` rotation matrices -> `(N,4)` [x,y,z,w].

    Shepperd's method: pick whichever of the four components is largest before
    taking a square root, so nothing is ever divided by something near zero. The
    sign of the whole quaternion is arbitrary (`q` and `-q` are the same
    rotation) and nothing downstream reads it -- the gravity target is quadratic
    in `q`, and the gate compares rotated vectors rather than components.
    """
    R = np.asarray(R, np.float64)
    m = lambda i, j: R[..., i, j]                                   # noqa: E731
    trace = m(0, 0) + m(1, 1) + m(2, 2)
    n = R.shape[0]
    out = np.zeros((n, 4), np.float64)

    big_w = trace > 0
    cand = np.stack([m(0, 0), m(1, 1), m(2, 2)], -1)
    big = np.argmax(cand, axis=-1)

    s = np.sqrt(np.maximum(trace + 1.0, 1e-12)) * 2.0
    out[big_w] = np.stack([(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s,
                           (m(1, 0) - m(0, 1)) / s, 0.25 * s], -1)[big_w]
    for k, (a, b, c) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        sel = (~big_w) & (big == k)
        if not sel.any():
            continue
        s = np.sqrt(np.maximum(1.0 + m(a, a) - m(b, b) - m(c, c), 1e-12)) * 2.0
        q = np.zeros((n, 4))
        q[:, a] = 0.25 * s
        q[:, b] = (m(b, a) + m(a, b)) / s
        q[:, c] = (m(a, c) + m(c, a)) / s
        q[:, 3] = (m(c, b) - m(b, c)) / s
        out[sel] = q[sel]
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- the draw

@dataclass(frozen=True)
class Remount:
    """One sample's augmentation: a constant map on body-frame vectors.

    `A` maps body components: `v_body -> A v_body`. It is a rotation when
    `mirrored` is False and a reflection (determinant -1) when it is True, and
    the gyroscope's sign follows the determinant because angular velocity is a
    pseudovector.
    """
    A: np.ndarray                 # (3,3) float64, the body-frame map
    mirrored: bool

    @property
    def gyro_sign(self) -> float:
        return -1.0 if self.mirrored else 1.0

    @property
    def world(self) -> np.ndarray:
        """The world-frame reflection that pairs with a body mirror; identity otherwise."""
        return WORLD_MIRROR if self.mirrored else np.eye(3)

    def vectors(self, v: np.ndarray) -> np.ndarray:
        """Body-frame true vectors: velocity, accelerometer, gravity direction."""
        return np.asarray(v, np.float64) @ self.A.T

    def pseudovectors(self, w: np.ndarray) -> np.ndarray:
        """Body-frame pseudovectors: the gyroscope, and nothing else here."""
        return self.gyro_sign * (np.asarray(w, np.float64) @ self.A.T)

    def channels(self, chan: np.ndarray, has_gravity: bool) -> np.ndarray:
        """`(F, C)` assembled channels -> the same channels, re-mounted.

        The layout is fixed by `prep.features.assemble`: accelerometer 0:3,
        gyroscope 3:6, gravity direction 6:9 when the representation carries it,
        then any extra scalars -- which are all norms, and so are untouched by a
        rotation by construction.
        """
        out = np.array(chan, np.float64, copy=True)
        out[:, 0:3] = self.vectors(out[:, 0:3])
        out[:, 3:6] = self.pseudovectors(out[:, 3:6])
        if has_gravity:
            out[:, 6:9] = self.vectors(out[:, 6:9])
        return out.astype(np.float32)

    def quaternions(self, q_xyzw: np.ndarray) -> np.ndarray:
        """`(K,4)` mid-window attitude -> the attitude of the re-mounted body.

        The new body frame's components satisfy `v' = A v`, so the attitude that
        carries them to the world is `R_world<-body A^T`, pre-multiplied by the
        world reflection when there is one. Both cases come out a proper
        rotation, which is why one matrix path serves them both.
        """
        q = np.asarray(q_xyzw, np.float64)
        valid = np.linalg.norm(q, axis=-1) > 1e-6         # zeros on padding / test
        if not valid.any():
            return np.asarray(q_xyzw, np.float32)
        R = quat_to_mat(q[valid])
        R_new = self.world @ R @ self.A.T
        out = np.array(q, np.float64, copy=True)
        out[valid] = mat_to_quat(R_new)
        return out.astype(np.float32)

    def positions(self, p: np.ndarray) -> np.ndarray:
        """World positions. A re-mounted sensor does not move the vehicle."""
        if not self.mirrored:
            return np.asarray(p, np.float32)
        return (np.asarray(p, np.float64) @ self.world.T).astype(np.float32)


IDENTITY = Remount(A=np.eye(3), mirrored=False)


# ------------------------------------------------------------------------ the sampler

@dataclass
class Augmenter:
    """Draws one `Remount` per sample, reproducibly.

    The draw depends on `(seed, epoch, chunk index)` and nothing else -- not on
    batch order, not on how many workers there are, not on which arm is
    training. That last one is deliberate: `m7_body`, `m15` and `m9_film`
    training on the same recipe and the same seed see the *identical* sequence
    of re-mountings, so a difference between those three arms is the model and
    never the augmentation.
    """
    yaw_deg: dict[str, float] = field(default_factory=dict)
    tilt_deg: float = 0.0
    mirror_p: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def enabled(self) -> bool:
        return (any(v > 0 for v in self.yaw_deg.values()) or self.tilt_deg > 0
                or any(v > 0 for v in self.mirror_p.values()))

    def draw(self, index: int, platform: str, acc_mean: np.ndarray) -> Remount:
        """The re-mounting for chunk `index` of platform `platform` this epoch.

        `acc_mean` is the chunk's mean accelerometer direction, which stands in
        for "which way is up" when choosing the yaw axis and the mirror plane.
        It is taken from the raw accelerometer rather than from the gravity
        channel on purpose: the six-channel arms have no gravity channel, and
        the three arms must see identical augmentation to be comparable. Its
        accuracy barely matters -- an axis a degree or two off simply means the
        yaw carries a degree or two of tilt with it, which is another legal
        re-mounting.
        """
        rng = np.random.default_rng([int(self.seed), int(self.epoch), int(index)])

        up = np.asarray(acc_mean, np.float64)
        n = np.linalg.norm(up)
        up = up / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

        A = np.eye(3)
        mirrored = False

        p_mirror = float(self.mirror_p.get(platform, 0.0))
        if p_mirror > 0 and rng.random() < p_mirror:
            # Reflect across the vertical plane through the body's forward axis.
            # Defined from `up` rather than from a fixed axis so it means the
            # same thing however the box is bolted on -- which matters, because
            # human's body +z sits 84 degrees from gravity.
            ex = np.array([1.0, 0.0, 0.0])
            e1 = ex - (ex @ up) * up
            if np.linalg.norm(e1) < 1e-6:            # forward is vertical; pick another
                ex = np.array([0.0, 1.0, 0.0])
                e1 = ex - (ex @ up) * up
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(up, e1)
            e2 /= np.linalg.norm(e2)
            A = (np.eye(3) - 2.0 * np.outer(e2, e2)) @ A
            mirrored = True

        yaw = float(self.yaw_deg.get(platform, 0.0))
        if yaw > 0:
            psi = rng.uniform(-np.radians(yaw), np.radians(yaw))
            A = rodrigues(up, psi) @ A

        if self.tilt_deg > 0:
            # A random axis perpendicular to `up`, so the op moves the tilt and
            # nothing else. This is TLIO's "gravity direction perturbation",
            # applied to the mounting rather than to the estimate.
            r = rng.normal(size=3)
            axis = r - (r @ up) * up
            if np.linalg.norm(axis) < 1e-9:
                axis = np.cross(up, [1.0, 0.0, 0.0])
            theta = rng.uniform(-np.radians(self.tilt_deg), np.radians(self.tilt_deg))
            A = rodrigues(axis, theta) @ A

        return Remount(A=A, mirrored=mirrored)


def build_augmenter(cfg) -> Augmenter | None:
    """The augmenter a config asks for, or None. Absent keys mean off.

    Following the M15 precedent in `config.py`: augmentation is switched on by
    override rather than declared in a reference, because every key sits inside
    the config hash and adding one to a reference would rehash every run already
    in `results.jsonl`.
    """
    aug = cfg.get("augment", {}) or {}
    ops = list(aug.get("ops", []) or [])
    if not ops:
        return None
    unknown = [o for o in ops if o not in ("remount", "timescale")]
    if unknown:
        raise KeyError(f"unknown augment op(s) {unknown}; 'remount' and 'timescale' are "
                       f"built. See .claude/augmentation.md for the designed-but-unbuilt tiers.")
    if "remount" not in ops:          # time scaling alone: data/timescale.py builds it
        return None
    spec = dict(aug.get("remount", {}) or {})
    recipe = spec.pop("recipe", None)
    if recipe is not None:
        if recipe not in RECIPES:
            raise KeyError(f"unknown augment recipe {recipe!r}; have {sorted(RECIPES)}")
        spec = {**RECIPES[recipe], **spec}
    yaw = {p: float(spec.get("yaw_deg", {}).get(p, 0.0)) for p in PLATFORMS}
    mir = {p: float(spec.get("mirror_p", {}).get(p, 0.0)) for p in PLATFORMS}
    a = Augmenter(yaw_deg=yaw, tilt_deg=float(spec.get("tilt_deg", 0.0)),
                  mirror_p=mir, seed=int(cfg["run"]["seed_augment"]))
    return a if a.enabled else None
