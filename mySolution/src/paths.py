"""Path resolution for everything under mySolution/.

Every path is derived from this file's location, so no script cares about the
working directory it was launched from. This mirrors repro/paths.py, which
serves the same purpose for the reproduction harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
MYSOLUTION = SRC.parent
ROOT = MYSOLUTION.parent

TARTAN = ROOT / "TartanIMU"                  # vendored, read-only
STARTER = TARTAN / "starter"
REPRO = MYSOLUTION / "repro"                 # the reproduction harness

RAW = MYSOLUTION / "data" / "raw"            # train/ val/ test/ index/
INDEX = RAW / "index"
CACHE = MYSOLUTION / "data" / "processed" / "cache"
RUNS = MYSOLUTION / "runs"
RESULTS_JSONL = RUNS / "results.jsonl"
SEAL_OPENINGS_JSONL = RUNS / "seal_openings.jsonl"

CACHE_VERSION = "v1"
CACHE_V = CACHE / CACHE_VERSION

WIN = 200                                    # frames per window (1.0 s)
FS = 200.0                                   # IMU rate, Hz
DT = WIN / FS                                # 1.0 s per window, always
PLATFORMS = ("car", "dog", "drone", "human")  # index == the NPZ platform_id
SPLITS = ("train", "val", "test")


def add_repro_to_path() -> None:
    """Make the vendored scorer and the repro harness importable.

    repro/score_val.py and repro/build_val_solution.py import each other by bare
    module name, so the repro directory itself has to be on sys.path, not just
    the starter directory.
    """
    for p in (str(STARTER), str(TARTAN), str(REPRO)):
        if p not in sys.path:
            sys.path.insert(0, p)


def traj_npz(split: str, platform: str | None, traj_id: str) -> Path:
    """Locate a trajectory's NPZ. test/ is flat; train/ and val/ nest by platform."""
    if split == "test":
        return RAW / "test" / f"{traj_id}.npz"
    return RAW / split / platform / f"{traj_id}.npz"
