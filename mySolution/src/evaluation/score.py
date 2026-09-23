"""Score a submission on a labelled split, with the per-platform breakdown.

The official scorer returns a single float. Because the metric macro-averages
over four platforms, that single float hides the only thing worth looking at:
**your worst platform is where your points are going.** This module reproduces
the scorer's internals to expose the breakdown, and then cross-checks itself
against `K.score()` on every call, so the breakdown can never drift away from
the number that actually counts.

The `solution` table is built from our own Tier 0 cache rather than re-derived
from the raw NPZ files, and `verify_solution_matches_repro()` asserts it is
identical to `repro/build_val_solution.py`'s. That does two jobs at once: it
makes the seal (carved out of *train*) scoreable with the same code path, and it
turns "does the cache carry the right q_gt and p_gt?" into a test instead of a
hope.

**ATE20 cannot be evaluated on test at all**, because test NPZ files carry no
`pos`, so no segment boundaries exist. Every ATE-based model-selection decision
must therefore go through val.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DT, PLATFORMS, add_repro_to_path
from ..prep.cache import load_cache

add_repro_to_path()
import kaggle_metric_tartanimu_score as K  # noqa: E402

SOLUTION_COLUMNS = ["window_id", "traj_id", "win_idx", "platform",
                    "qx", "qy", "qz", "qw", "gx", "gy", "gz", "dt",
                    "vx_gt", "vy_gt", "vz_gt"]


@lru_cache(maxsize=4)
def build_solution(split: str = "val") -> pd.DataFrame:
    """The scorer's `solution` table for a labelled split, from the Tier 0 cache.

    Conventions, pinned in CLAUDE.md finding 3 and re-verified against the
    published ~0.151 m ATE20 floor: position at the window END, quaternion at
    the window MIDDLE, target the per-window MEAN of `vel_body`. Do not change
    them silently -- every other pairing is 3 to 8 times worse.
    """
    c = load_cache(split)
    if not c.has_labels:
        raise ValueError(f"split {split!r} carries no labels, so it cannot be scored. "
                         f"Test NPZ files have no `pos`, so ATE20 has no segments.")
    traj = c.trajectories
    sol = pd.DataFrame({
        "window_id": c.window_id,
        "traj_id": traj.traj_id.to_numpy()[c.traj_idx],
        "win_idx": c.win_idx,
        "platform": traj.platform.to_numpy()[c.traj_idx],
    })
    sol[["qx", "qy", "qz", "qw"]] = np.asarray(c.q_gt)   # [x, y, z, w], scalar-last
    sol[["gx", "gy", "gz"]] = np.asarray(c.p_gt)
    sol["dt"] = DT
    sol[["vx_gt", "vy_gt", "vz_gt"]] = np.asarray(c.v_gt)
    return sol[SOLUTION_COLUMNS]


def verify_solution_matches_repro(atol: float = 0.0) -> None:
    """Assert the cache-built val solution equals repro/build_val_solution.py's."""
    add_repro_to_path()
    from build_val_solution import build as repro_build

    ours = build_solution("val").sort_values("window_id").reset_index(drop=True)
    theirs = repro_build("end", "mid").sort_values("window_id").reset_index(drop=True)
    for col in ("window_id", "traj_id", "win_idx", "platform"):
        # Values, not dtypes: the cache stores win_idx as int32 to halve its size.
        assert np.array_equal(ours[col].to_numpy(), theirs[col].to_numpy()), \
            f"solution column {col} differs"
    num = ["qx", "qy", "qz", "qw", "gx", "gy", "gz", "dt", "vx_gt", "vy_gt", "vz_gt"]
    a, b = ours[num].to_numpy(np.float64), theirs[num].to_numpy(np.float64)
    worst = float(np.abs(a - b).max())
    assert worst <= atol, (f"cache-built solution differs from repro's by {worst:.3e} "
                           f"(tolerance {atol:.3e})")


def score_submission(submission: pd.DataFrame | str | Path, split: str = "val",
                     cross_check: bool = True, traj_ids=None) -> dict:
    """Return AVE, ATE20 and the TartanIMU Score, per platform and macro-averaged.

    `traj_ids` restricts the scoring to a named set of trajectories. That is what
    makes the **sealed holdout** scoreable: the seal is 61 trajectories carved
    out of the *train* split, so scoring it means scoring a subset of that split
    -- through this same function, and therefore through the same official
    scorer cross-check, as val. Without the restriction the merge below would
    demand predictions for all of train.
    """
    if isinstance(submission, (str, Path)):
        submission = pd.read_csv(submission)
    sol = build_solution(split)
    if traj_ids is not None:
        wanted = set(str(t) for t in traj_ids)
        sol = sol[sol.traj_id.astype(str).isin(wanted)].reset_index(drop=True)
        if sol.empty:
            raise ValueError(f"none of the {len(wanted)} requested trajectories are "
                             f"in split {split!r}")

    s = submission.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"})
    merged = sol.merge(s[["window_id", "vx_pred", "vy_pred", "vz_pred"]],
                       on="window_id", how="left")
    if merged[["vx_pred", "vy_pred", "vz_pred"]].isna().any().any():
        raise ValueError("submission does not cover every window in the solution")

    per: dict[str, list[tuple[float, float]]] = {}
    dropped = 0
    for _, g in merged.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        ate, ave = K._ate_traj(g), K._ave_traj(g)
        if np.isfinite(ate) and np.isfinite(ave):
            per.setdefault(g["platform"].iloc[0], []).append((ate, ave))
        else:
            dropped += 1

    platforms = {}
    for p in PLATFORMS:
        if p not in per:
            continue
        ates = [x[0] for x in per[p]]
        aves = [x[1] for x in per[p]]
        platforms[p] = {"n_traj": len(ates),
                        "ave": float(np.mean(aves)),
                        "ate20": float(np.mean(ates))}
    macro_ate = float(np.mean([v["ate20"] for v in platforms.values()]))
    macro_ave = float(np.mean([v["ave"] for v in platforms.values()]))
    score = K.W_AVE * (macro_ave / K.AVE_REF) + K.W_ATE * (macro_ate / K.ATE_REF)

    out = {"split": split, "platforms": platforms, "macro_ave": macro_ave,
           "macro_ate20": macro_ate, "score": score,
           "trajectories_dropped": dropped}
    if cross_check:
        official = float(K.score(sol.copy(), submission.copy(), "window_id"))
        out["official_score"] = official
        out["agrees_with_official"] = bool(abs(official - score) < 1e-9)
        if not out["agrees_with_official"]:
            raise AssertionError(f"per-platform reconstruction {score:.6f} disagrees "
                                 f"with the official scorer {official:.6f}")
    return out


def format_report(result: dict, name: str = "") -> str:
    lines = [f"=== {name or result['split']} ===",
             f"{'platform':10s} {'n_traj':>6s} {'AVE (m/s)':>11s} {'ATE20 (m)':>11s}"]
    for p, v in result["platforms"].items():
        lines.append(f"{p:10s} {v['n_traj']:6d} {v['ave']:11.4f} {v['ate20']:11.4f}")
    lines.append(f"{'MACRO':10s} {'':6s} {result['macro_ave']:11.4f} "
                 f"{result['macro_ate20']:11.4f}")
    lines.append(f"TartanIMU Score = 0.6*({result['macro_ave']:.4f}/{K.AVE_REF:.4f})"
                 f" + 0.4*({result['macro_ate20']:.4f}/{K.ATE_REF:.4f})"
                 f" = {result['score']:.4f}")
    if "official_score" in result:
        tag = "match" if result["agrees_with_official"] else "MISMATCH"
        lines.append(f"official scorer call            = "
                     f"{result['official_score']:.4f}  [{tag}]")
    return "\n".join(lines)
