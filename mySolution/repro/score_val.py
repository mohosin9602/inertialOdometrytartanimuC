#!/usr/bin/env python3
"""Self-score a submission on the labelled val split using the official scorer.

Reports the AVE / ATE20 components and the per-platform breakdown -- the scorer
itself returns only a single float, but the macro-average over four platforms
means the worst platform is where your points are going.

Usage:
    python score_val.py                       # floor + zeros + baseline
    python score_val.py out/submission_val.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from paths import OUT, add_import_paths

add_import_paths()
import kaggle_metric_tartanimu_score as K   # noqa: E402
from build_val_solution import build        # noqa: E402

SOL = build(pos_at="end", quat_at="mid")


def components(sub):
    """Reproduce the scorer's internals to expose what the single float hides."""
    s = sub.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"})
    m = SOL.merge(s, on="window_id", how="left")
    per = {}
    for _, g in m.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        ate, ave = K._ate_traj(g), K._ave_traj(g)
        if np.isfinite(ate) and np.isfinite(ave):
            per.setdefault(g["platform"].iloc[0], []).append((ate, ave))
    rows = {p: (float(np.mean([x[0] for x in v])),
                float(np.mean([x[1] for x in v])), len(v))
            for p, v in sorted(per.items())}
    macro_ate = float(np.mean([r[0] for r in rows.values()]))
    macro_ave = float(np.mean([r[1] for r in rows.values()]))
    score = K.W_AVE * (macro_ave / K.AVE_REF) + K.W_ATE * (macro_ate / K.ATE_REF)
    return rows, macro_ate, macro_ave, score


def report(name, sub):
    rows, macro_ate, macro_ave, score = components(sub)
    official = K.score(SOL.copy(), sub.copy(), "window_id")
    print(f"\n=== {name} ===")
    print(f"{'platform':10s} {'n_traj':>6s} {'AVE (m/s)':>11s} {'ATE20 (m)':>11s}")
    for p, (ate, ave, n) in rows.items():
        print(f"{p:10s} {n:6d} {ave:11.4f} {ate:11.4f}")
    print(f"{'MACRO':10s} {'':6s} {macro_ave:11.4f} {macro_ate:11.4f}")
    print(f"TartanIMU Score = 0.6*({macro_ave:.4f}/{K.AVE_REF:.4f}) "
          f"+ 0.4*({macro_ate:.4f}/{K.ATE_REF:.4f}) = {score:.4f}")
    agree = "match" if abs(official - score) < 1e-9 else "MISMATCH"
    print(f"official scorer call            = {official:.4f}  [{agree}]")
    return score


def main():
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            report(os.path.basename(path), pd.read_csv(path))
        return

    gt = SOL[["window_id", "vx_gt", "vy_gt", "vz_gt"]].rename(
        columns={"vx_gt": "vx", "vy_gt": "vy", "vz_gt": "vz"})
    report("ground-truth velocities (floor)", gt)

    z = SOL[["window_id"]].copy()
    z[["vx", "vy", "vz"]] = 0.0
    report("all zeros", z)

    baseline = os.path.join(OUT, "submission_val.csv")
    if os.path.exists(baseline):
        report("released unified baseline (routed)", pd.read_csv(baseline))

    forced = os.path.join(OUT, "submission_val_headhuman.csv")
    if os.path.exists(forced):
        report("released baseline, --head human forced", pd.read_csv(forced))


if __name__ == "__main__":
    main()
