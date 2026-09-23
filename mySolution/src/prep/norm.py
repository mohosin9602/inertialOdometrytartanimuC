"""Tier 0 -- fixed input normalisation constants, computed once over train.

    mySolution/.venv/bin/python -m src.prep.norm                # build for body + body_grav
    mySolution/.venv/bin/python -m src.prep.norm --show

**This module exists because of one specific, silent way the model can fail.**

Neural networks train better when their inputs are roughly centred and roughly
unit-scale. The obvious way to arrange that is to standardise each window by its
own mean and variance. **Do not do that here.** What distinguishes a drone from
a walking human is precisely amplitude and frequency structure -- a drone
vibrates hard and fast, a human sways gently and slowly. Rescaling every window
to the same variance deletes exactly that difference, and it deletes it
*silently*: the loss still falls, the gradients still flow, and the embodiment
branch simply never learns what it was built to learn.

So the constants are computed **once, over the whole train split**, written to
disk, and thereafter loaded. Every window is divided by the same number, so the
loud windows stay loud and the quiet ones stay quiet. `models.md` §5 names this
as the one concrete way the embodiment design fails, and `tests/test_m5.py`
asserts the constants are loaded rather than recomputed.

**Statistics are per channel, over frames.** The six IMU channels have genuinely
different units and scales -- accelerometer around 10 m/s^2 because gravity is
in there, gyroscope around 0.2 rad/s -- so one shared scale would leave the
gyroscope invisible. The three gravity channels are already unit vectors and
come out with a scale near 1 by construction, which is a useful sanity check
that the file was built from the right array.

**Trajectories are weighted equally, not by length.** A single 774,081-frame dog
recording holds more frames than every drone flight put together, so a plain
frame-weighted mean would be a statement about that one recording. Averaging
per-trajectory statistics first, then over platforms, matches how the metric
itself averages and keeps the constants representative of all four platforms.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import CACHE_V, PLATFORMS
from ..utils import read_json, write_json
from .cache import load_cache
from .features import assemble, needs_gravity
from .orientation import load_orientation, orientation_available

#: Which representations to precompute constants for when run with no arguments.
DEFAULT_REPRS = ("body", "body_grav")

#: A channel whose measured spread is below this is treated as constant and
#: given a scale of 1.0, so dividing by it can never explode.
MIN_STD = 1e-6


def norm_path(input_repr: str, root: Path | None = None) -> Path:
    return (root or CACHE_V) / "norm" / f"{input_repr}.json"


@dataclass(frozen=True)
class NormStats:
    """Per-channel centre and scale, applied as `(x - mean) / std`."""

    input_repr: str
    mean: np.ndarray        # (C,) float32
    std: np.ndarray         # (C,) float32
    source: str             # where these came from, for the run record

    def apply(self, channels: np.ndarray) -> np.ndarray:
        """`(F, C)` raw channels -> `(F, C)` standardised, float32."""
        if channels.shape[1] != len(self.mean):
            raise ValueError(f"{self.input_repr} constants cover {len(self.mean)} "
                             f"channels but got {channels.shape[1]}")
        return ((channels - self.mean) / self.std).astype(np.float32)

    def as_record(self) -> dict:
        return {"input_repr": self.input_repr, "source": self.source,
                "mean": [round(float(v), 6) for v in self.mean],
                "std": [round(float(v), 6) for v in self.std]}


def identity_stats(input_repr: str, n_channels: int) -> NormStats:
    """The `normalisation: none` option -- leaves the input exactly as it is."""
    return NormStats(input_repr, np.zeros(n_channels, np.float32),
                     np.ones(n_channels, np.float32), source="identity")


# --------------------------------------------------------------------------- building

def compute(input_repr: str, split: str = "train", root: Path | None = None) -> NormStats:
    """Measure per-channel mean and standard deviation, trajectories weighted equally.

    Two passes over the data: the first collects each trajectory's own mean and
    mean-of-squares, the second combines them. Pooling the squares rather than
    the standard deviations is what makes the combination exact -- averaging
    standard deviations directly would understate the spread.
    """
    cache = load_cache(split)
    gravity = (load_orientation(split, root) if needs_gravity(input_repr) else None)

    per_platform_mean: dict[str, list[np.ndarray]] = {}
    per_platform_sq: dict[str, list[np.ndarray]] = {}

    for t in range(cache.n_traj):
        row = cache.trajectories.iloc[t]
        lo = int(row.frame_offset)
        hi = lo + int(row.n_frames_kept)
        frames = np.asarray(cache.imu[lo:hi], np.float32)
        grav = (np.asarray(gravity[lo:hi], np.float32) if gravity is not None else None)
        channels = assemble(frames, input_repr, gravity=grav).astype(np.float64)

        platform = str(row.platform)
        per_platform_mean.setdefault(platform, []).append(channels.mean(axis=0))
        per_platform_sq.setdefault(platform, []).append((channels ** 2).mean(axis=0))

    # Average within each platform, then across platforms -- the same hierarchy
    # the score itself uses, so no platform's constants are drowned out.
    platform_mean = np.stack([np.mean(per_platform_mean[p], axis=0) for p in PLATFORMS])
    platform_sq = np.stack([np.mean(per_platform_sq[p], axis=0) for p in PLATFORMS])
    mean = platform_mean.mean(axis=0)
    mean_sq = platform_sq.mean(axis=0)

    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std = np.maximum(np.sqrt(variance), MIN_STD)
    return NormStats(input_repr, mean.astype(np.float32), std.astype(np.float32),
                     source=f"{split}:trajectory_then_platform_mean")


def build(input_repr: str, split: str = "train", root: Path | None = None,
          force: bool = False, verbose: bool = True) -> NormStats:
    path = norm_path(input_repr, root)
    if path.exists() and not force:
        if verbose:
            print(f"[{input_repr}] norm constants exist, skipping "
                  f"(use --force to rebuild)")
        return load(input_repr, root)
    stats = compute(input_repr, split, root)
    write_json(path, stats.as_record())
    if verbose:
        print(f"[{input_repr}] wrote {path.name}")
        print(_report(stats))
    return stats


def load(input_repr: str, root: Path | None = None) -> NormStats:
    """Read the committed constants. Never recomputes -- that is the whole point."""
    path = norm_path(input_repr, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no normalisation constants for input_repr={input_repr!r} at {path}.\n"
            f"They are measured once over train and then fixed forever; per-window "
            f"standardisation is not an acceptable substitute (see this module's "
            f"docstring). Build them with:\n"
            f"    mySolution/.venv/bin/python -m src.prep.norm")
    d = read_json(path)
    return NormStats(d["input_repr"], np.asarray(d["mean"], np.float32),
                     np.asarray(d["std"], np.float32), source=d["source"])


def _report(stats: NormStats) -> str:
    names = (["ax", "ay", "az", "gx", "gy", "gz"] +
             (["ux", "uy", "uz"] if len(stats.mean) == 9 else []))
    lines = [f"    {'channel':<8} {'mean':>10} {'std':>10}"]
    for i, m in enumerate(stats.mean):
        name = names[i] if i < len(names) else f"c{i}"
        lines.append(f"    {name:<8} {m:>10.4f} {stats.std[i]:>10.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-repr", action="append", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--show", action="store_true", help="print existing constants")
    a = ap.parse_args()

    reprs = a.input_repr or [r for r in DEFAULT_REPRS
                             if not needs_gravity(r) or orientation_available(a.split)]
    for r in reprs:
        if a.show:
            print(f"[{r}]\n{_report(load(r))}")
        else:
            build(r, a.split, force=a.force)


if __name__ == "__main__":
    main()
