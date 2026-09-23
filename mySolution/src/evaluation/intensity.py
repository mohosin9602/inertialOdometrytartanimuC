"""How hard is each window moving -- and how good is a submission on the hard half?

`.claude/augmentation.md` §17. Test's drones move harder than train's (linear
acceleration 1.87x), while val's move more gently (0.75x), so val's plain drone
AVE under-rewards anything aimed at hard flying. The readout for time scaling
therefore splits val's drone windows at their median intensity and scores each
half the way the metric scores a platform: error per window, averaged per
trajectory, then over trajectories.

**Intensity** is the RMS, over a window's 200 frames, of the LINEAR acceleration
below 5 Hz: the accelerometer low-passed, minus box5's gravity estimate. It reads
only the IMU, so it exists on every split, and it is exactly the quantity a
speed-up scales by s^2. Vibration (above 10 Hz) is left out on purpose -- it is
what separates the platforms, not how hard they are moving.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.timescale import G0, band
from ..paths import FS, WIN
from ..prep.cache import load_cache
from ..prep.orientation import gravity_box


def window_intensity(split: str = "val", platform: str = "drone") -> pd.DataFrame:
    """One row per window of `platform` on a labelled split: id, trajectory, intensity."""
    c = load_cache(split)
    rows = c.trajectories
    out = []
    for t in range(c.n_traj):
        if str(rows.platform.iloc[t]) != platform:
            continue
        r = rows.iloc[t]
        lo, n, nw = int(r.frame_offset), int(r.n_frames_kept), int(r.n_windows)
        frames = np.asarray(c.imu[lo:lo + n], np.float64)
        lin = band(frames[:, :3], 0.0, 5.0) - G0 * gravity_box(frames, 1.0 / FS, 5.0)
        rms = np.sqrt((lin ** 2).sum(1).reshape(nw, WIN).mean(1))
        wsl = c.windows_slice(t, 0, nw)
        out.append(pd.DataFrame({"window_id": np.asarray(c.window_id[wsl]),
                                 "traj_id": str(r.traj_id), "intensity": rms,
                                 "vx": c.v_gt[wsl][:, 0], "vy": c.v_gt[wsl][:, 1],
                                 "vz": c.v_gt[wsl][:, 2]}))
    return pd.concat(out, ignore_index=True)


def subset_ave(errors: pd.DataFrame) -> float:
    """Metric-style AVE of a set of windows: mean per trajectory, then over trajectories."""
    return float(errors.groupby("traj_id").err.mean().mean())


def halves(submission, table: pd.DataFrame | None = None, split: str = "val",
           platform: str = "drone") -> dict:
    """AVE of `platform` on its calmer and harder halves (split at the median intensity).

    `submission` is a path or a frame with `window_id, vx, vy, vz`. Pass `table`
    (from `window_intensity`) to score several submissions without recomputing it.
    """
    if table is None:
        table = window_intensity(split, platform)
    sub = pd.read_csv(submission) if isinstance(submission, (str, Path)) else submission
    m = table.merge(sub[["window_id", "vx", "vy", "vz"]], on="window_id",
                    suffixes=("", "_pred"), how="left")
    if m[["vx_pred", "vy_pred", "vz_pred"]].isna().any().any():
        raise ValueError("the submission does not cover every window of this platform")
    m["err"] = np.linalg.norm(m[["vx_pred", "vy_pred", "vz_pred"]].to_numpy()
                              - m[["vx", "vy", "vz"]].to_numpy(), axis=1)
    cut = float(table.intensity.median())
    return {"all": subset_ave(m), "calm": subset_ave(m[m.intensity <= cut]),
            "intense": subset_ave(m[m.intensity > cut]), "cut": cut}
