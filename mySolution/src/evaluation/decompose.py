"""Direction-versus-magnitude decomposition of velocity error, per platform.

This is M9 arm 0 — the zero-GPU diagnostic that gates the `direction_speed`
head (arm 5). Finding 8 measured this decomposition on the **released
baseline** and found that on car, dog and drone the model had essentially no
directional information (median angle 80-90 degrees). That was a platform
-identity failure, which is precisely what the internal embodiment branch was
built to fix. The question arm 0 asks is whether it *stayed* fixed: if drone's
residual error is now magnitude-dominated, a head that predicts direction and
speed separately is answering a question the model no longer has.

**The decomposition.** For a predicted velocity `p` and a true velocity `g`,
the squared error splits exactly into two non-negative parts::

    |p - g|^2  =  (|p| - |g|)^2  +  2|p||g|(1 - cos t)
      E_total       E_mag                E_dir

`E_mag` is the error that would remain if the direction were perfect; `E_dir`
is the extra error the wrong direction costs. The identity is exact, so the
two fractions always sum to one, and both terms stay well defined when `|g|`
is zero (a stationary window is pure magnitude error, which is the honest
reading).

This is the same decomposition Finding 8 used, and pointing this module at the
submission that finding was measured from returns that finding's table: median
angle and mean cosine to the last published digit on all four platforms, dog's
"direction 94%" at 94.1%, drone's "magnitude 75%" at 74.6%. `tests/test_m9.py`
pins that reproduction, which is what makes the rest of the output here worth
believing.

**Why a fraction is not enough.** A percentage says where the squared error
sits, not what removing it is worth — and the competition score is a weighted
sum of a *mean of norms* and a trajectory-integration term, neither of which
is a sum of squares. So this module also builds two oracle counterfactuals and
scores them through the real scorer:

    direction oracle — keep the predicted speed, take the true direction
    magnitude oracle — keep the predicted direction, take the true speed
    traj_bias oracle — subtract each trajectory's own mean error

Applied to one platform at a time, those give the answer in score points, the
same currency Finding 6 used to retire zero-velocity detection at 0.0058. The
oracles are not additive: fixing direction and magnitude together would leave
zero error, so the individual gains generally sum past what is really there.

**An oracle is a ceiling, not a forecast.** It answers "is there enough here to
be worth chasing", and a small number is therefore much more decisive than a
large one — 0.0077 means stop, while 0.0535 only means the target is real, not
that any particular component will reach it.

**This is a diagnostic and reads ground truth by construction.** It lives in
`evaluation/` for that reason — Tier 2 (`post/`) is forbidden from touching a
label, and an oracle is not a post-processing op. Nothing here may ever run on
the inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import PLATFORMS, RUNS
from .score import build_solution, score_submission

PRED_COLUMNS = ["vx", "vy", "vz"]
GT_COLUMNS = ["vx_gt", "vy_gt", "vz_gt"]

# A window slower than this carries almost no directional information: its
# heading is dominated by sensor noise, and Finding 6 measured that windows
# below it are worth almost nothing to the metric anyway (a window whose true
# speed is 0.02 m/s can contribute at most 0.02 m/s of error). Angle statistics
# are reported on the moving subset only; the energy decomposition uses every
# window, because it stays well defined at zero.
MOVING_THRESHOLD = 0.05


def _norms(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def merge_with_truth(submission: pd.DataFrame | str | Path,
                     split: str = "val") -> pd.DataFrame:
    """Join a submission to the solution table, keeping platform and truth."""
    if isinstance(submission, (str, Path)):
        submission = pd.read_csv(submission)
    sol = build_solution(split)
    merged = sol.merge(submission[["window_id"] + PRED_COLUMNS],
                       on="window_id", how="left")
    if merged[PRED_COLUMNS].isna().any().any():
        raise ValueError("submission does not cover every window in the solution")
    return merged


def direction_oracle(pred: np.ndarray, gt: np.ndarray,
                     eps: float = 1e-12) -> np.ndarray:
    """Keep the predicted speed, take the true direction.

    Where the true velocity is exactly zero there is no direction to copy, so
    the prediction is left alone — the remaining error is pure magnitude and no
    directional oracle could remove it.
    """
    gn = _norms(gt)
    ok = gn > eps
    out = pred.astype(np.float64, copy=True)
    out[ok] = gt[ok] / gn[ok, None] * _norms(pred)[ok, None]
    return out


def magnitude_oracle(pred: np.ndarray, gt: np.ndarray,
                     eps: float = 1e-12) -> np.ndarray:
    """Keep the predicted direction, take the true speed.

    Where the prediction is exactly zero there is no direction to rescale, so
    it is left alone — the remaining error is pure direction.
    """
    pn = _norms(pred)
    ok = pn > eps
    out = pred.astype(np.float64, copy=True)
    out[ok] = pred[ok] / pn[ok, None] * _norms(gt)[ok, None]
    return out


def traj_bias_oracle(pred: np.ndarray, gt: np.ndarray,
                     traj: np.ndarray) -> np.ndarray:
    """Subtract each trajectory's own mean error — a constant offset per run.

    This is the ceiling on what any purely per-trajectory correction could buy,
    which is the quantity M9 arm 2 is really asking about: a wider embodiment
    vector cannot add information, but it can carry within-platform variation,
    and a constant offset per trajectory is the simplest form that takes.
    """
    out = pred.astype(np.float64, copy=True)
    err = pred - gt
    for tid in np.unique(traj):
        k = traj == tid
        out[k] -= err[k].mean(axis=0)
    return out


ORACLES = {"direction": direction_oracle, "magnitude": magnitude_oracle}
GROUPED_ORACLES = {"traj_bias": traj_bias_oracle}


def oracle_submission(submission: pd.DataFrame | str | Path, mode: str,
                      split: str = "val",
                      platforms: tuple[str, ...] | None = None) -> pd.DataFrame:
    """A counterfactual submission with one oracle applied to some platforms.

    `platforms=None` applies it everywhere. Restricting to a single platform is
    what turns the diagnostic into a score-point statement about that platform,
    since the metric macro-averages and the other three are held fixed.
    """
    if mode not in ORACLES and mode not in GROUPED_ORACLES:
        raise ValueError(f"unknown oracle {mode!r}, have "
                         f"{sorted({*ORACLES, *GROUPED_ORACLES})}")
    merged = merge_with_truth(submission, split)
    pred = merged[PRED_COLUMNS].to_numpy(np.float64)
    gt = merged[GT_COLUMNS].to_numpy(np.float64)
    if mode in GROUPED_ORACLES:
        fixed = GROUPED_ORACLES[mode](pred, gt, merged["traj_id"].to_numpy())
    else:
        fixed = ORACLES[mode](pred, gt)

    if platforms is not None:
        keep = ~merged["platform"].isin(platforms).to_numpy()
        fixed[keep] = pred[keep]
    out = pd.DataFrame({"window_id": merged["window_id"].to_numpy()})
    out[PRED_COLUMNS] = fixed
    return out


def decompose(submission: pd.DataFrame | str | Path, split: str = "val",
              moving_threshold: float = MOVING_THRESHOLD) -> dict:
    """Per-platform direction/magnitude geometry of the velocity error."""
    merged = merge_with_truth(submission, split)
    pred = merged[PRED_COLUMNS].to_numpy(np.float64)
    gt = merged[GT_COLUMNS].to_numpy(np.float64)
    platform = merged["platform"].to_numpy()

    pn, gn = _norms(pred), _norms(gt)
    e_total = _norms(pred - gt) ** 2
    e_mag = (pn - gn) ** 2
    # Exact by the identity; clamped because catastrophic cancellation can push
    # it a few ulp below zero when the direction is very nearly right.
    e_dir = np.maximum(e_total - e_mag, 0.0)

    out: dict[str, dict] = {}
    for p in PLATFORMS:
        m = platform == p
        if not m.any():
            continue
        moving = m & (gn > moving_threshold) & (pn > 1e-12)
        cos = np.clip(np.sum(pred[moving] * gt[moving], axis=1)
                      / (pn[moving] * gn[moving]), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        tot = float(e_total[m].sum())
        out[p] = {
            "n_windows": int(m.sum()),
            "moving_frac": float(moving.sum() / m.sum()),
            "median_angle_deg": float(np.median(angle)) if moving.any() else float("nan"),
            "mean_cos": float(cos.mean()) if moving.any() else float("nan"),
            # Pooled, not a mean of per-window ratios: a per-window ratio blows
            # up as the true speed approaches zero, which is exactly where the
            # ratio means least.
            "speed_ratio": float(pn[m].sum() / gn[m].sum()),
            "frac_dir": float(e_dir[m].sum() / tot) if tot > 0 else float("nan"),
            "frac_mag": float(e_mag[m].sum() / tot) if tot > 0 else float("nan"),
        }
        out[p]["dominant"] = ("direction" if out[p]["frac_dir"] >= 0.5
                              else "magnitude")
    return out


def bias_structure(submission: pd.DataFrame | str | Path,
                   split: str = "val") -> dict:
    """Split each platform's error into a per-trajectory constant and a residual.

    Finding 8 measured that per-trajectory bias norms far exceed platform-pooled
    ones (drone 0.78 against 0.25), meaning the correlated bias ATE20 punishes is
    largely *per-trajectory* rather than a property of the platform as a whole.
    That distinction is what M9 arm 2 turns on: identifying which of four
    platforms a window came from is saturated at 1.000 accuracy, so if a large
    share of the residual is a constant offset that differs *between*
    trajectories of the same platform, the embodiment vector needs room for
    within-platform variation and width is a live knob.

    `bias_frac_sq_err` is the share of squared error a per-trajectory oracle
    offset would remove — the ceiling on what any purely per-trajectory
    correction could buy.
    """
    merged = merge_with_truth(submission, split)
    pred = merged[PRED_COLUMNS].to_numpy(np.float64)
    gt = merged[GT_COLUMNS].to_numpy(np.float64)
    err = pred - gt
    platform = merged["platform"].to_numpy()
    traj = merged["traj_id"].to_numpy()

    out: dict[str, dict] = {}
    for p in PLATFORMS:
        m = platform == p
        if not m.any():
            continue
        e, t = err[m], traj[m]
        norms, removed, total = [], 0.0, float((e ** 2).sum())
        for tid in np.unique(t):
            k = t == tid
            b = e[k].mean(axis=0)
            norms.append(float(np.linalg.norm(b)))
            removed += float((e[k] ** 2).sum() - ((e[k] - b) ** 2).sum())
        pooled = float(np.linalg.norm(e.mean(axis=0)))
        traj_bias = float(np.mean(norms))
        out[p] = {
            "traj_bias_norm": traj_bias,
            "pooled_bias_norm": pooled,
            "bias_ratio": traj_bias / pooled if pooled > 0 else float("nan"),
            "bias_frac_sq_err": removed / total if total > 0 else float("nan"),
        }
    return out


def oracle_gains(submission: pd.DataFrame | str | Path,
                 split: str = "val") -> dict:
    """What each oracle, applied to one platform alone, is worth in score points."""
    base = score_submission(submission, split, cross_check=False)
    gains: dict[str, dict] = {"baseline": base}
    for p in PLATFORMS:
        if p not in base["platforms"]:
            continue
        row: dict[str, float] = {}
        for mode in (*ORACLES, *GROUPED_ORACLES):
            sub = oracle_submission(submission, mode, split, platforms=(p,))
            r = score_submission(sub, split, cross_check=False)
            row[f"{mode}_score"] = r["score"]
            row[f"{mode}_gain"] = base["score"] - r["score"]
            row[f"{mode}_ave"] = r["platforms"][p]["ave"]
            row[f"{mode}_ate20"] = r["platforms"][p]["ate20"]
        gains[p] = row
    return gains


# --------------------------------------------------------------- presentation

def format_decomposition(result: dict, name: str = "") -> str:
    lines = [f"=== direction / magnitude decomposition — {name or 'val'} ===",
             f"{'platform':9s} {'n_win':>7s} {'moving':>7s} {'med.ang':>8s} "
             f"{'meancos':>8s} {'spd_r':>7s} {'E_dir':>7s} {'E_mag':>7s}  dominant"]
    for p, v in result.items():
        lines.append(
            f"{p:9s} {int(v['n_windows']):7d} {v['moving_frac']:7.1%} "
            f"{v['median_angle_deg']:7.1f}° {v['mean_cos']:8.3f} "
            f"{v['speed_ratio']:7.3f} {v['frac_dir']:7.1%} {v['frac_mag']:7.1%}  "
            f"{v['dominant']}")
    return "\n".join(lines)


def format_bias(result: dict, name: str = "") -> str:
    """How much of the error is a constant offset per trajectory."""
    lines = [f"=== error structure: constant bias vs residual — {name or 'val'} ===",
             f"{'platform':9s} {'|bias| per traj':>15s} {'|bias| pooled':>14s} "
             f"{'ratio':>7s} {'bias share of sq.err':>21s}"]
    for p, v in result.items():
        lines.append(
            f"{p:9s} {v['traj_bias_norm']:15.3f} {v['pooled_bias_norm']:14.3f} "
            f"{v['bias_ratio']:7.2f} {v['bias_frac_sq_err']:20.1%}")
    return "\n".join(lines)


def format_gains(gains: dict, name: str = "") -> str:
    base = gains["baseline"]
    lines = [f"=== perfect-oracle value, one platform at a time — {name or 'val'} ===",
             f"    baseline score {base['score']:.4f}   "
             f"(macro AVE {base['macro_ave']:.4f}, ATE20 {base['macro_ate20']:.4f})",
             f"{'platform':9s} {'AVE now':>8s} | {'dir AVE':>8s} {'Δscore':>8s} "
             f"| {'mag AVE':>8s} {'Δscore':>8s} | {'bias AVE':>8s} {'Δscore':>8s}"]
    for p in PLATFORMS:
        if p not in gains:
            continue
        v, b = gains[p], base["platforms"][p]
        lines.append(
            f"{p:9s} {b['ave']:8.4f} | {v['direction_ave']:8.4f} "
            f"{v['direction_gain']:8.4f} | {v['magnitude_ave']:8.4f} "
            f"{v['magnitude_gain']:8.4f} | {v['traj_bias_ave']:8.4f} "
            f"{v['traj_bias_gain']:8.4f}")
    return "\n".join(lines)


def aggregate(rows: list[dict]) -> dict:
    """Mean and range across seeds, for every numeric leaf of a nested table.

    Non-numeric leaves are carried through from the first row, and a `_spread`
    sibling is added beside every numeric one — the spread is the whole point,
    because a conclusion that flips between seeds is not a conclusion. Drone's
    own 2-sigma is 0.0208 against the aggregate's 0.0037, so its spread is the
    one to read.
    """
    def merge(vals: list):
        first = vals[0]
        if isinstance(first, dict):
            return {k: v for key in first
                    for k, v in _entry(key, [r[key] for r in vals]).items()}
        return first

    def _entry(key: str, vals: list) -> dict:
        if isinstance(vals[0], dict):
            return {key: merge(vals)}
        if isinstance(vals[0], bool) or not isinstance(vals[0], (int, float)):
            return {key: vals[0]}
        a = np.asarray(vals, dtype=np.float64)
        return {key: float(a.mean()), key + "_spread": float(a.max() - a.min())}

    return merge(rows)


# ----------------------------------------------------------------------- CLI

def _resolve(target: str) -> tuple[str, Path]:
    """Accept a run directory, a run-directory name, or a submission CSV."""
    p = Path(target)
    if p.is_file():
        return p.stem, p
    if not p.is_dir():
        p = RUNS / target
    if p.is_dir():
        csv = p / "predictions" / "submission_val.csv"
        if not csv.is_file():
            raise SystemExit(f"{p} has no predictions/submission_val.csv")
        return p.name, csv
    raise SystemExit(f"cannot resolve {target!r} as a run or a submission csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("targets", nargs="+",
                    help="run directories (or names, or submission CSVs)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--oracles", action="store_true",
                    help="also score the direction and magnitude oracles")
    ap.add_argument("--per-run", action="store_true",
                    help="print each run's table as well as the aggregate")
    ap.add_argument("--json", type=Path, help="write the full result here")
    args = ap.parse_args()

    resolved = [_resolve(t) for t in args.targets]
    decs, bias, gains, blob = [], [], [], {}
    solo = args.per_run or len(resolved) == 1
    for name, csv in resolved:
        d = decompose(csv, args.split)
        b = bias_structure(csv, args.split)
        decs.append(d)
        bias.append(b)
        blob.setdefault(name, {}).update(decomposition=d, bias_structure=b)
        if solo:
            print(format_decomposition(d, name), "\n")
            print(format_bias(b, name), "\n")
        if args.oracles:
            g = oracle_gains(csv, args.split)
            gains.append(g)
            blob[name]["oracle_gains"] = g
            if solo:
                print(format_gains(g, name), "\n")

    if len(resolved) > 1:
        label = f"{len(resolved)} runs, mean"
        print(format_decomposition(aggregate(decs), label), "\n")
        print(format_bias(aggregate(bias), label), "\n")
        blob["aggregate"] = {"decomposition": aggregate(decs),
                             "bias_structure": aggregate(bias),
                             "runs": [n for n, _ in resolved]}
        if gains:
            print(format_gains(aggregate(gains), label), "\n")
            blob["aggregate"]["oracle_gains"] = aggregate(gains)

    if args.json:
        args.json.write_text(json.dumps(blob, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
