"""Reading side of the Tier 0 window cache.

**Storage layout differs from tensor layout, deliberately** (PipelinePlan.md
§2.3). On disk the IMU is *frame-major and flat* — `(total_frames, C)` — so a
chunk of K consecutive windows is exactly one contiguous byte range,
`imu[start*200 : (start+K)*200]`. The reshape to `(K, C, 200)` happens at
collate time on a copy already small enough to sit in L2. A window-major layout
would need K strided reads instead of one.

Frames are truncated to whole windows: a trajectory of `n` frames keeps
`floor(n/200)*200` of them, which is exactly the span the `floor(n/200)` windows
in the index CSV cover.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import CACHE_V, PLATFORMS, WIN

# Channel order inside the cached IMU array. Verified against the raw NPZ:
# columns 0:3 have mean norm 10.0 m/s^2 (gravity is present), columns 3:6 have
# mean norm 0.17 rad/s.
CHANNELS_BODY = ("ax", "ay", "az", "gx", "gy", "gz")

ARRAYS_ALWAYS = ("imu", "window_id", "traj_idx", "win_idx")
ARRAYS_LABELLED = ("quat_gt", "v_gt", "q_gt", "p_gt")


def split_dir(split: str, root: Path | None = None) -> Path:
    return (root or CACHE_V) / split


def cache_exists(split: str, root: Path | None = None) -> bool:
    d = split_dir(split, root)
    return (d / "meta.json").exists() and (d / "imu.npy").exists()


@dataclass
class SplitCache:
    """Memory-mapped view of one split's cache.

    Loading is `mmap_mode="r"`, so opening a split costs microseconds and the
    pages the dataset actually touches are the only ones read. Measured random
    chunk reads: 0.022-0.031 ms warm. Data loading is free relative to any
    training step, which is why zero dataloader workers is the right default.
    """

    split: str
    root: Path
    meta: dict
    trajectories: pd.DataFrame     # one row per trajectory, indexed by traj_idx
    imu: np.ndarray                # (F, C) float16 memmap, frame-major flat
    window_id: np.ndarray          # (W,) int64
    traj_idx: np.ndarray           # (W,) int32
    win_idx: np.ndarray            # (W,) int32
    quat_gt: np.ndarray | None     # (F, 4) float16 per-frame, train/val only
    v_gt: np.ndarray | None        # (W, 3) float64 per-window mean of vel_body
    q_gt: np.ndarray | None        # (W, 4) float32 mid-window quaternion [x,y,z,w]
    p_gt: np.ndarray | None        # (W, 3) float32 end-window position, world

    @property
    def has_labels(self) -> bool:
        return self.v_gt is not None

    @property
    def n_windows(self) -> int:
        return int(self.window_id.shape[0])

    @property
    def n_traj(self) -> int:
        return int(len(self.trajectories))

    @property
    def n_channels(self) -> int:
        return int(self.imu.shape[1])

    def traj_row(self, t: int) -> pd.Series:
        return self.trajectories.iloc[t]

    def frames(self, t: int, start_win: int, n_win: int) -> np.ndarray:
        """One contiguous frame block: `(n_win*200, C)` float16, no copy."""
        off = int(self.trajectories.frame_offset.iloc[t]) + start_win * WIN
        return self.imu[off: off + n_win * WIN]

    def windows_slice(self, t: int, start_win: int, n_win: int) -> slice:
        """Row range in the per-window arrays for `n_win` windows from `start_win`."""
        off = int(self.trajectories.window_offset.iloc[t]) + start_win
        return slice(off, off + n_win)


def _maybe(path: Path) -> np.ndarray | None:
    return np.load(path, mmap_mode="r") if path.exists() else None


def load_cache(split: str, root: Path | None = None) -> SplitCache:
    d = split_dir(split, root)
    if not cache_exists(split, root):
        raise FileNotFoundError(
            f"no window cache for split {split!r} at {d}. "
            f"Build it with:  mySolution/.venv/bin/python -m src.prep.windows")
    import json
    meta = json.loads((d / "meta.json").read_text())
    traj = pd.read_csv(d / "trajectories.csv")
    return SplitCache(
        split=split,
        root=d,
        meta=meta,
        trajectories=traj,
        imu=np.load(d / "imu.npy", mmap_mode="r"),
        window_id=np.load(d / "window_id.npy"),
        traj_idx=np.load(d / "traj_idx.npy"),
        win_idx=np.load(d / "win_idx.npy"),
        quat_gt=_maybe(d / "quat_gt.npy"),
        v_gt=_maybe(d / "v_gt.npy"),
        q_gt=_maybe(d / "q_gt.npy"),
        p_gt=_maybe(d / "p_gt.npy"),
    )


def platform_id(name: str) -> int:
    """`car`->0, `dog`->1, `drone`->2, `human`->3, matching the NPZ platform_id."""
    return PLATFORMS.index(name)
