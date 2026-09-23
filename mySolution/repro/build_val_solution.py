#!/usr/bin/env python3
"""Build a scorer-compatible `solution` table for the labelled val split.

The starter kit ships no such builder, but kaggle_metric_tartanimu_score.score()
requires columns:
    window_id, traj_id, win_idx, platform, qx,qy,qz,qw, gx,gy,gz, dt,
    vx_gt,vy_gt,vz_gt

Two conventions are undocumented and materially change ATE20:
  * which sample of `pos`  each window reports as its ground-truth position
  * which sample of `quat` rotates that window's body velocity into the world

Rather than guess, they are pinned against a published invariant: feeding the
ground-truth velocities back through the scorer must reproduce the ~0.151 m
ATE20 floor. `pos=window end, quat=window mid` gives 0.0997 m on val; every
other pairing is 3-8x worse. That is also the theoretically correct pairing --
the scorer builds P = cumsum(R @ v * dt), so P[i] is the position at the *end*
of window i, and v is the *mean* body velocity over the window, best rotated by
the mid-window attitude.

Run directly to re-verify the calibration:  python build_val_solution.py
"""
import os

import numpy as np
import pandas as pd

from paths import DATA, FS, WIN


def build(pos_at="end", quat_at="mid"):
    """Return the solution dataframe the official scorer consumes."""
    win = pd.read_csv(os.path.join(DATA, "index", "val_windows.csv"))
    tgt = pd.read_csv(os.path.join(DATA, "index", "val_targets.csv")).rename(
        columns={"vx": "vx_gt", "vy": "vy_gt", "vz": "vz_gt"})
    df = win.merge(tgt, on="window_id", how="left")
    assert df[["vx_gt", "vy_gt", "vz_gt"]].notna().all().all(), "missing targets"

    rows = []
    for (platform, traj_id), g in df.groupby(["platform", "traj_id"], sort=False):
        npz = np.load(os.path.join(DATA, "val", platform, f"{traj_id}.npz"))
        pos, quat = npz["pos"], npz["quat"]
        n = pos.shape[0]
        k = g["win_idx"].to_numpy()

        if pos_at == "end":
            pi = np.minimum((k + 1) * WIN, n - 1)
        elif pos_at == "mid":
            pi = np.minimum(k * WIN + WIN // 2, n - 1)
        else:
            pi = np.minimum(k * WIN, n - 1)

        if quat_at == "mid":
            qi = np.minimum(k * WIN + WIN // 2, n - 1)
        elif quat_at == "start":
            qi = np.minimum(k * WIN, n - 1)
        else:
            qi = np.minimum((k + 1) * WIN, n - 1)

        out = g.copy()
        out[["gx", "gy", "gz"]] = pos[pi]
        out[["qx", "qy", "qz", "qw"]] = quat[qi]   # npz quat is (x, y, z, w)
        out["dt"] = WIN / FS
        rows.append(out)

    sol = pd.concat(rows, ignore_index=True)
    return sol[["window_id", "traj_id", "win_idx", "platform",
                "qx", "qy", "qz", "qw", "gx", "gy", "gz", "dt",
                "vx_gt", "vy_gt", "vz_gt"]]


def _calibrate():
    """Re-derive the convention choice from the published GT-velocity floor."""
    from paths import add_import_paths
    add_import_paths()
    import kaggle_metric_tartanimu_score as K

    npz = np.load(os.path.join(DATA, "val", "car", "car_val_0000.npz"))
    tgt = pd.read_csv(os.path.join(DATA, "index", "val_targets.csv"))
    w = pd.read_csv(os.path.join(DATA, "index", "val_windows.csv"))
    w0 = w[(w.traj_id == "car_val_0000") & (w.win_idx == 0)].window_id.iloc[0]
    print("published target[win 0]  =", tgt[tgt.window_id == w0][["vx", "vy", "vz"]].to_numpy()[0])
    print("mean(vel_body[0:200])    =", npz["vel_body"][:WIN].mean(0))
    print()

    print("GT-velocity ATE20 by convention (published floor on test = 0.151 m):")
    best = None
    for pos_at in ("end", "mid", "start"):
        for quat_at in ("mid", "start", "end"):
            sol = build(pos_at, quat_at)
            gt = sol[["window_id", "vx_gt", "vy_gt", "vz_gt"]].rename(
                columns={"vx_gt": "vx", "vy_gt": "vy", "vz_gt": "vz"})
            s = K.score(sol.copy(), gt.copy(), "window_id")
            ate = s / K.W_ATE * K.ATE_REF     # AVE is exactly 0 here
            print(f"  pos={pos_at:5s} quat={quat_at:5s} -> ATE20 {ate:.4f} m")
            if best is None or ate < best[0]:
                best = (ate, pos_at, quat_at)
    print(f"\nbest: pos={best[1]}, quat={best[2]} at {best[0]:.4f} m")


if __name__ == "__main__":
    _calibrate()
