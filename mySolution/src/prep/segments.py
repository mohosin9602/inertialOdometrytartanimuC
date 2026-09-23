"""Tier 0 -- the ATE20 segment cache. PipelinePlan.md §0.5, §6.1, §9 (M1).

    mySolution/.venv/bin/python -m src.prep.segments            # train + val, ~2 s
    mySolution/.venv/bin/python -m src.prep.segments --force

ATE20 chops a trajectory's ground-truth path into pieces about 20 metres long,
aligns each piece to the truth on its own, and measures how far off it lands.
Those pieces -- the *segments* -- depend only on ground-truth position, so they
never change during training and are worth computing once. This module computes
them and writes 3,042 rows to disk.

**It calls the vendored `_segment_bounds`, never a reimplementation.** The
cutting rule has three fiddly parts (a distance-based cut, a short-tail merge, a
short-head fold) and any of them drifting would make the training loss quietly
disagree with the scorer. Importing the real function makes drift impossible,
and `tests/test_m1.py` re-checks the cache against a live call on all 475
trajectories anyway.

**The one surprise, and it is easy to get wrong.** Segments tile the travelled
*distance* without overlap, but their *index sets overlap by one window* at
every internal boundary: the scorer sets `start = end` and then slices
`[s:e+1]` inclusive. So if segment 0 is windows 0..30 then segment 1 starts at
window 30, and **window 30 belongs to both**, contributing to two segment
errors and accumulating gradient twice. That is correct behaviour to reproduce.
Any accounting that assumes each window contributes exactly once is wrong.

That shared window also shapes the per-window `seg_id` this module hands the
dataloader. One integer per window cannot say "this window is in two segments",
so the convention is: **a window carries the id of the last segment that starts
at or before it.** A shared boundary window therefore carries the *later*
segment's id, and the earlier segment is exactly `{w : seg_id[w] == j}` plus one
extra window on the right. Nothing needs to remember that rule by hand -- ask
`SegmentCache.contained()` for the true inclusive bounds, which is what the
ATE20 loss will use at M10.

**ATE20 has no segments on test.** Test NPZ files carry no `pos`, so there is no
path to cut. This cache exists for train and val only, which is also why every
ATE-based model-selection decision has to go through val.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import CACHE_V, PLATFORMS, add_repro_to_path
from ..utils import env_record, write_json
from .cache import load_cache

add_repro_to_path()
import kaggle_metric_tartanimu_score as K  # noqa: E402

LABELLED_SPLITS = ("train", "val")


def segments_dir(root: Path | None = None) -> Path:
    return (root or CACHE_V) / "segments"


# --------------------------------------------------------------------------- building

def _bounds_for_trajectory(positions: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) window pairs for one trajectory's ground-truth path.

    `positions` is (n_windows, 3) in float64 -- the same array the scorer sees,
    since it reads the solution table's gx/gy/gz columns and casts to float.
    """
    return K._segment_bounds(np.asarray(positions, np.float64))


def build_split(split: str, root: Path | None = None, force: bool = False,
                verbose: bool = True) -> dict:
    """Cut every trajectory in one labelled split into segments and cache the bounds."""
    out = segments_dir(root)
    path = out / f"{split}.npz"
    if path.exists() and not force:
        if verbose:
            print(f"[{split}] segment cache exists, skipping (use --force to rebuild)")
        return {}

    cache = load_cache(split)
    if not cache.has_labels:
        raise ValueError(
            f"split {split!r} has no ground-truth position, so it has no ATE20 "
            f"segments. Test NPZ files carry no `pos`; this is a fact about the "
            f"competition, not a missing file.")

    t0 = time.perf_counter()
    traj_idx, starts, ends = [], [], []

    for t in range(cache.n_traj):
        n_windows = int(cache.trajectories.n_windows.iloc[t])
        positions = cache.p_gt[cache.windows_slice(t, 0, n_windows)]
        for s, e in _bounds_for_trajectory(positions):
            traj_idx.append(t)
            starts.append(s)
            ends.append(e)

    traj_idx = np.asarray(traj_idx, np.int32)
    starts = np.asarray(starts, np.int32)
    ends = np.asarray(ends, np.int32)
    platform_of_traj = cache.trajectories.platform.to_numpy()
    per_platform = {p: int((platform_of_traj[traj_idx] == p).sum()) for p in PLATFORMS}

    out.mkdir(parents=True, exist_ok=True)
    np.savez(path, traj_idx=traj_idx, start=starts, end=ends)

    meta = {
        "split": split,
        "n_segments": int(len(starts)),
        "n_trajectories": int(cache.n_traj),
        "per_platform": per_platform,
        "trajectories_with_one_segment": int(
            (np.bincount(traj_idx, minlength=cache.n_traj) == 1).sum()),
        "segment_length_m": K.SEGMENT_LENGTH_M,
        "min_segment_points": K.MIN_SEGMENT_POINTS,
        "build_seconds": round(time.perf_counter() - t0, 2),
        "env": env_record(),
    }
    write_json(out / f"{split}_meta.json", meta)
    if verbose:
        print(f"[{split}] {meta['n_segments']} segments over {cache.n_traj} "
              f"trajectories in {meta['build_seconds']}s -> {path}")
    return meta


# --------------------------------------------------------------------------- reading

@dataclass
class SegmentCache:
    """All of one split's segments, plus the lookups the dataloader needs.

    The three arrays are parallel and sorted by trajectory, so `first` can hold
    row offsets and every per-trajectory lookup is a slice rather than a search.
    A segment's **global id is simply its row number**, which is what ends up in
    the batch's `seg_id` key.
    """

    split: str
    traj_idx: np.ndarray     # (S,) int32  which trajectory each segment belongs to
    start: np.ndarray        # (S,) int32  first window index, inclusive
    end: np.ndarray          # (S,) int32  last window index, inclusive
    first: np.ndarray        # (n_traj+1,) int64 row offsets into the arrays above

    @property
    def n_segments(self) -> int:
        return int(len(self.start))

    def rows_of(self, t: int) -> slice:
        """Row range holding trajectory `t`'s segments."""
        return slice(int(self.first[t]), int(self.first[t + 1]))

    def bounds_of(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(global_ids, starts, ends)` for one trajectory. Ends are inclusive."""
        r = self.rows_of(t)
        return np.arange(r.start, r.stop), self.start[r], self.end[r]

    def starts_of(self, t: int) -> np.ndarray:
        return self.start[self.rows_of(t)]

    def contained(self, t: int, chunk_start: int, chunk_len: int):
        """The segments of `t` that fit **entirely** inside this chunk.

        A segment straddling a chunk edge is excluded from the ATE20 term
        altogether -- half a segment cannot be aligned against the truth. The
        velocity term still trains on those windows, and the exclusion rate is
        logged per platform (`ChunkDataset.segment_coverage`).
        """
        ids, s, e = self.bounds_of(t)
        fits = (s >= chunk_start) & (e <= chunk_start + chunk_len - 1)
        return ids[fits], s[fits], e[fits]

    def window_seg_id(self, t: int, chunk_start: int, chunk_len: int) -> np.ndarray:
        """Per-window segment id for one chunk; `-1` where no whole segment covers it.

        Segments are written in order, so a boundary window shared by segments
        `j` and `j+1` ends up carrying `j+1` -- the convention described in this
        module's docstring.
        """
        out = np.full(chunk_len, -1, np.int64)
        ids, s, e = self.contained(t, chunk_start, chunk_len)
        for seg_id, a, b in zip(ids, s, e):
            out[a - chunk_start: b - chunk_start + 1] = seg_id
        return out


def load_segments(split: str, root: Path | None = None) -> SegmentCache:
    path = segments_dir(root) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no segment cache for split {split!r} at {path}. Build it with:\n"
            f"    mySolution/.venv/bin/python -m src.prep.segments")
    d = np.load(path)
    traj_idx = d["traj_idx"]
    n_traj = load_cache(split).n_traj
    # searchsorted over a sorted array turns "which rows belong to trajectory t"
    # into two integer lookups instead of a scan.
    first = np.searchsorted(traj_idx, np.arange(n_traj + 1), side="left").astype(np.int64)
    first[n_traj] = len(traj_idx)
    return SegmentCache(split=split, traj_idx=traj_idx, start=d["start"],
                        end=d["end"], first=first)


def segments_available(split: str, root: Path | None = None) -> bool:
    return (segments_dir(root) / f"{split}.npz").exists()


# --------------------------------------------------------------------------- entrypoint

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=LABELLED_SPLITS, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    total = 0
    for split in ([a.split] if a.split else LABELLED_SPLITS):
        meta = build_split(split, force=a.force)
        total += meta.get("n_segments", 0)
    if total:
        print(f"{total} segments cached in total")


if __name__ == "__main__":
    main()
