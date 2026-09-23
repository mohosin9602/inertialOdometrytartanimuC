"""Tier 0 -- the sealed holdout and the 5-fold grouped assignment. PipelinePlan.md §6.6.

    mySolution/.venv/bin/python -m src.prep.splits            # draw once, then verify
    mySolution/.venv/bin/python -m src.prep.splits --force    # redraw (almost never)

Two things live here, and both answer the same question: *which trajectories is
a run allowed to look at?*

**The seal.** `val` is 80 trajectories and will be tuned against a hundred times
over the next three weeks. The seal is how we find out whether any of that
tuning was real -- a slice carved out of *train* that no training run and no
model-selection decision ever touches, opened three times in the whole project,
each opening written to an audit log.

**The five folds.** Anything claimed in the report gets a grouped
cross-validation number. Grouping is by *trajectory*, never by window, because
two windows one second apart are almost the same sample; splitting them across a
fold boundary would measure memorisation rather than generalisation.

Two numbers to hold in mind before reading any seal result (§6.6):

* The seal's noise floor is **SE ~= 0.0155**. A val-versus-seal gap smaller than
  about **0.03 is not evidence of anything**.
* This seed-42 draw is measurably about one sigma *harder* than the train
  population. **Never read the seal's absolute score.** Compare seal to seal
  across model versions, where that offset cancels out.

**The selection rule, fixed here and never re-picked.** Per platform: take the
trajectories whose window count falls inside that platform's [p25, p75] band,
permute them with seed 42, keep the first N. Drawing from the middle of the
length distribution is what stops the seal from being accidentally made of only
the shortest or only the longest recordings. The per-platform N comes from
§6.6's sizing argument -- each platform contributes roughly equally to the
standard error of the seal's macro score, which is the right notion of
"balanced" for a metric that macro-averages over platforms.

**The draw is committed, not recomputed on demand.** `configs/splits_v1.json` is
a tracked file. This module can redraw it, but the default is to redraw into
memory and *check* it still matches the file, so a change to the data or to this
rule fails loudly instead of silently moving the seal underneath a half-finished
ablation table.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import MYSOLUTION, PLATFORMS, SEAL_OPENINGS_JSONL
from ..utils import append_jsonl, read_json, write_json
from .cache import load_cache

NAME = "tartanimu_sealed_holdout_v1"
SEED = 42
N_FOLDS = 5

#: Per-platform seal size, from the §6.6 sizing argument. Do not "round" these:
#: human at 5 rather than 2 is what drops the whole seal's standard error from
#: 0.0192 to 0.0155, and human is the only platform that can afford the tax.
SEAL_SIZES = {"car": 6, "dog": 8, "drone": 42, "human": 5}

SPLITS_FILE = MYSOLUTION / "configs" / "splits_v1.json"


# --------------------------------------------------------------------------- drawing

def _middle_half(rows: pd.DataFrame) -> pd.DataFrame:
    """One platform's trajectories whose window count sits in its [p25, p75] band."""
    lo, hi = np.percentile(rows.n_windows.to_numpy(), [25, 75])
    return rows[(rows.n_windows >= lo) & (rows.n_windows <= hi)]


def draw(trajectories: pd.DataFrame) -> dict:
    """Draw the seal and the folds from the train trajectory table.

    One random generator, seeded once, consumed in a fixed platform order, so
    the whole draw is a pure function of the trajectory table.
    """
    rng = np.random.default_rng(SEED)
    seal: list[str] = []
    sizing: dict[str, dict] = {}

    for platform in PLATFORMS:
        # sort_values makes the draw independent of how the table happened to be built
        rows = trajectories[trajectories.platform == platform].sort_values("traj_id")
        band = _middle_half(rows)
        want = SEAL_SIZES[platform]
        if len(band) < want:
            raise ValueError(
                f"{platform}: only {len(band)} trajectories sit in the [p25, p75] "
                f"window-count band, but the §6.6 sizing table asks for {want}")
        picked = sorted(str(t) for t in rng.permutation(band.traj_id.to_numpy())[:want])
        seal += picked
        sizing[platform] = {
            "sealed": want,
            "of": int(len(rows)),
            "eligible": int(len(band)),
            "sealed_windows": int(rows[rows.traj_id.isin(picked)].n_windows.sum()),
            "platform_windows": int(rows.n_windows.sum()),
        }

    # The folds cover what is left of train. Stratification by platform falls out
    # of dealing each platform's trajectories round-robin into the five folds.
    rest = trajectories[~trajectories.traj_id.isin(seal)]
    folds: dict[str, int] = {}
    for platform in PLATFORMS:
        ids = sorted(rest[rest.platform == platform].traj_id.astype(str))
        for i, traj_id in enumerate(rng.permutation(np.asarray(ids, dtype=object))):
            folds[str(traj_id)] = i % N_FOLDS

    total_windows = int(trajectories.n_windows.sum())
    sealed_windows = sum(s["sealed_windows"] for s in sizing.values())
    return {
        "name": NAME,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "rule": ("per platform: trajectories whose window count is inside that "
                 "platform's [p25, p75] band, permuted with seed 42, first N kept"),
        "seal": sorted(seal),
        "folds": folds,
        "sizing": sizing,
        "sealed_trajectories": len(seal),
        "sealed_windows": sealed_windows,
        "sealed_window_fraction": round(sealed_windows / total_windows, 5),
    }


# --------------------------------------------------------------------------- reading

@dataclass(frozen=True)
class Splits:
    """The committed draw, plus the two lookups a dataset actually needs."""

    name: str
    seed: int
    n_folds: int
    seal: tuple[str, ...]
    folds: dict[str, int]

    def _rows(self, cache, traj_ids) -> list[int]:
        """Trajectory ids -> row numbers in this cache's trajectory table."""
        wanted = set(traj_ids)
        return [i for i, t in enumerate(cache.trajectories.traj_id.astype(str))
                if t in wanted]

    def seal_rows(self, cache) -> list[int]:
        """Rows to hand to `ChunkDataset(traj_subset=...)` to score the seal."""
        return self._rows(cache, self.seal)

    def fold_rows(self, cache, fold: int, role: str = "train") -> list[int]:
        """Rows for one cross-validation fold. `role` is "train" or "val"."""
        if role not in ("train", "val"):
            raise ValueError(f"role must be 'train' or 'val', not {role!r}")
        keep = {t for t, f in self.folds.items()
                if (f != fold if role == "train" else f == fold)}
        return self._rows(cache, keep)


def load_splits(path: Path | None = None) -> Splits:
    d = read_json(path or SPLITS_FILE)
    return Splits(name=d["name"], seed=d["seed"], n_folds=d["n_folds"],
                  seal=tuple(d["seal"]), folds={k: int(v) for k, v in d["folds"].items()})


# --------------------------------------------------------------------- seal audit trail

def record_opening(reason: str, score: float | None = None, **extra) -> None:
    """Append one line to `runs/seal_openings.jsonl`.

    The seal is worth exactly as much as the discipline around it, and that
    discipline is easier to keep when every opening leaves a trace. §6.6 budgets
    three openings for the whole project: M4, M12, M13.
    """
    append_jsonl(SEAL_OPENINGS_JSONL, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seal": NAME, "reason": reason, "score": score, **extra})


# --------------------------------------------------------------------------- entrypoint

def build(force: bool = False, verbose: bool = True) -> dict:
    """Draw the splits. If the file already exists, check the draw still matches it."""
    fresh = draw(load_cache("train").trajectories)

    if SPLITS_FILE.exists() and not force:
        committed = read_json(SPLITS_FILE)
        if committed["seal"] != fresh["seal"] or committed["folds"] != fresh["folds"]:
            raise SystemExit(
                f"the draw no longer reproduces {SPLITS_FILE}.\nThe seal is committed "
                f"and must never move, so something changed in the trajectory table or "
                f"in the selection rule. Investigate before reaching for --force.")
        if verbose:
            print(f"{SPLITS_FILE.name}: verified, the draw still reproduces it")
        return committed

    write_json(SPLITS_FILE, fresh)
    if verbose:
        print(f"wrote {SPLITS_FILE}")
    return fresh


def _report(d: dict) -> str:
    lines = [f"{d['name']}  seed {d['seed']}  {d['n_folds']} folds",
             f"{'platform':<9} {'sealed':>7} {'of':>5} {'eligible':>9} {'% windows':>10}"]
    for p in PLATFORMS:
        s = d["sizing"][p]
        pct = 100 * s["sealed_windows"] / s["platform_windows"]
        lines.append(f"{p:<9} {s['sealed']:>7} {s['of']:>5} {s['eligible']:>9} {pct:>9.1f}%")
    lines.append(f"total     {d['sealed_trajectories']:>7} trajectories, "
                 f"{d['sealed_windows']:,} windows "
                 f"({100 * d['sealed_window_fraction']:.1f}% of train)")
    fold_sizes = {f: sum(1 for v in d["folds"].values() if v == f)
                  for f in range(d["n_folds"])}
    lines.append(f"folds     {fold_sizes} trajectories")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="redraw and overwrite the committed file")
    a = ap.parse_args()
    print(_report(build(force=a.force)))


if __name__ == "__main__":
    main()
