"""Write `window_id, vx, vy, vz` -- and refuse to write a broken one.

PipelinePlan.md §2.6 step 5. The assertions are the point: a submission that
fails any of them is not written at all, because a silently short or duplicated
file costs a Kaggle attempt and, worse, produces a number that looks like a
result. The expected id list comes from the Tier 0 cache, which was itself built
from `index/<split>_windows.csv`, so the check is against the organizers' own
index rather than against whatever the model happened to produce.

The submission contains velocity only. You never submit orientation: the
quaternion, ground-truth position, platform and `dt` columns belong to the
scorer's *solution* table, which the organizers hold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ("window_id", "vx", "vy", "vz")
EXPECTED_ROWS = {"train": 81931, "val": 23714, "test": 30644}


def check_submission(table: pd.DataFrame, expected_ids: np.ndarray,
                     split: str | None = None) -> None:
    """Raise on anything that would make this file invalid. No side effects."""
    missing_cols = [c for c in COLUMNS if c not in table.columns]
    if missing_cols:
        raise ValueError(f"submission is missing column(s) {missing_cols}")

    expected = np.asarray(expected_ids, np.int64)
    if split is not None and split in EXPECTED_ROWS:
        if len(expected) != EXPECTED_ROWS[split]:
            raise ValueError(f"the {split} index has {len(expected)} windows but "
                             f"{EXPECTED_ROWS[split]} were expected -- the cache and "
                             f"the raw index disagree")
    if len(table) != len(expected):
        raise ValueError(f"submission has {len(table)} rows, expected {len(expected)}")

    ids = table["window_id"].to_numpy(np.int64)
    if pd.Series(ids).duplicated().any():
        dupes = pd.Series(ids)[pd.Series(ids).duplicated()].unique()[:5]
        raise ValueError(f"duplicate window_id(s): {dupes}")
    if not np.array_equal(np.sort(ids), np.sort(expected)):
        missing = np.setdiff1d(expected, ids)
        extra = np.setdiff1d(ids, expected)
        raise ValueError(f"window_id set mismatch: {len(missing)} missing "
                         f"{missing[:5]}, {len(extra)} unexpected {extra[:5]}")

    v = table[["vx", "vy", "vz"]].to_numpy(np.float64)
    if not np.isfinite(v).all():
        bad = int((~np.isfinite(v)).any(axis=1).sum())
        raise ValueError(f"{bad} rows contain a non-finite velocity")


def write_submission(table: pd.DataFrame, path: str | Path,
                     expected_ids: np.ndarray, split: str | None = None) -> Path:
    """Validate, then write. Nothing is written if a check fails."""
    check_submission(table, expected_ids, split)
    out = table[list(COLUMNS)].sort_values("window_id").reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
