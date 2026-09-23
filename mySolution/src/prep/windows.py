"""Tier 0 — build the flat, frame-major window cache. PipelinePlan.md §2.3, §6.4.

    mySolution/.venv/bin/python -m src.prep.windows            # all three splits
    mySolution/.venv/bin/python -m src.prep.windows --split val --force

What it writes, per split, under data/processed/cache/v1/<split>/:

    imu.npy          (F, 6) float16   frame-major flat, truncated to whole windows
    quat_gt.npy      (F, 4) float16   per-frame ground-truth attitude (train/val)
    window_id.npy    (W,)   int64     the submission row id
    traj_idx.npy     (W,)   int32     row in trajectories.csv
    win_idx.npy      (W,)   int32     window position inside its trajectory
    v_gt.npy         (W, 3) float64   target: per-window mean of vel_body
    q_gt.npy         (W, 4) float32   mid-window quaternion [x, y, z, w]
    p_gt.npy         (W, 3) float32   end-window position, world frame
    trajectories.csv                  one row per trajectory
    meta.json                         provenance and counts

Three things are deliberate:

* **fp16 on disk, fp32 at collate.** Measured over every train frame: zero
  overflows, worst per-window RMS error 3.0e-4 m/s^2, about 0.00025 score
  points. `--dtype f32` exists so that claim stays cheap to re-test.
* **The per-frame quaternion is cached, not a gravity unit vector.** A gravity
  vector pins roll and pitch but carries no heading, so a rotation cannot be
  recovered from it without inventing a yaw. Storing the quaternion keeps the
  tilt-alignment experiment (input_repr=`aligned`) and the learned-attitude door
  open for 161 MB. This array is ground truth and is for diagnostics and
  augmentation checks only -- `prep.orientation` will write its own filtered
  quaternion alongside it, with a `source` field, and the training entry point
  refuses a `quat`-sourced cache unless run.diagnostic_only is set.
* **q_gt and p_gt use the pinned sampling convention** -- position at the window
  END, quaternion at the window MIDDLE (CLAUDE.md finding 3). The indexing here
  is byte-identical to repro/build_val_solution.py, including its clamp to
  `n - 1`, so a cache-built solution table and the scorer's cannot drift apart.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import CACHE_V, INDEX, PLATFORMS, SPLITS, WIN, traj_npz
from ..utils import env_record, write_json
from .cache import CHANNELS_BODY, split_dir

DTYPES = {"f16": np.float16, "f32": np.float32}


def _index_table(split: str) -> pd.DataFrame:
    w = pd.read_csv(INDEX / f"{split}_windows.csv")
    if "platform" not in w.columns:           # test is anonymised
        w["platform"] = ""
    return w


def _targets(split: str) -> pd.DataFrame | None:
    p = INDEX / f"{split}_targets.csv"
    return pd.read_csv(p) if p.exists() else None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_split(split: str, dtype: str = "f16", root: Path | None = None,
                force: bool = False, verbose: bool = True) -> dict:
    """Build one split's cache. Returns the meta record it wrote."""
    out = split_dir(split, root)
    if out.exists() and not force and (out / "meta.json").exists():
        if verbose:
            print(f"[{split}] cache exists, skipping (use --force to rebuild)")
        import json
        return json.loads((out / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    np_dtype = DTYPES[dtype]

    w = _index_table(split)
    tgt = _targets(split)
    labelled = tgt is not None
    if labelled:
        assert np.array_equal(w.window_id.to_numpy(), tgt.window_id.to_numpy()), \
            f"{split}_targets.csv is not row-aligned with {split}_windows.csv"

    groups = list(w.groupby("traj_id", sort=False))
    n_windows_total = len(w)
    n_frames_total = n_windows_total * WIN

    # Preallocate memory-mapped outputs; each trajectory is written once, in order.
    imu_mm = np.lib.format.open_memmap(
        out / "imu.npy", mode="w+", dtype=np_dtype, shape=(n_frames_total, 6))
    quat_mm = (np.lib.format.open_memmap(
        out / "quat_gt.npy", mode="w+", dtype=np_dtype, shape=(n_frames_total, 4))
        if labelled else None)

    window_id = np.zeros(n_windows_total, np.int64)
    traj_idx = np.zeros(n_windows_total, np.int32)
    win_idx = np.zeros(n_windows_total, np.int32)
    # float64, not float32: the published targets CSV is the authority and is
    # float64 text. Storing it narrower makes the cache-built solution table
    # differ from repro/build_val_solution.py at ~9e-16, which is harmless but
    # costs an exact-equality check that is worth 2.5 MB to keep.
    v_gt = np.zeros((n_windows_total, 3), np.float64) if labelled else None
    q_gt = np.zeros((n_windows_total, 4), np.float32) if labelled else None
    p_gt = np.zeros((n_windows_total, 3), np.float32) if labelled else None

    rows, frame_off, win_off = [], 0, 0
    worst_target_gap = 0.0
    t0 = time.perf_counter()

    for t, (traj_id, g) in enumerate(groups):
        platform = str(g.platform.iloc[0])
        npz = np.load(traj_npz(split, platform or None, traj_id))
        imu = npz["imu"]
        n_raw = imu.shape[0]
        k = g.win_idx.to_numpy()
        n_win = len(g)
        assert np.array_equal(k, np.arange(n_win)), f"{traj_id}: win_idx is not 0..n-1"
        assert n_raw // WIN == n_win, \
            f"{traj_id}: floor({n_raw}/{WIN}) = {n_raw // WIN} but the index says {n_win}"

        keep = n_win * WIN
        imu_mm[frame_off: frame_off + keep] = imu[:keep].astype(np_dtype)

        sl = slice(win_off, win_off + n_win)
        window_id[sl] = g.window_id.to_numpy()
        traj_idx[sl] = t
        win_idx[sl] = k

        pid = -1
        if labelled:
            pos, quat, vel = npz["pos"], npz["quat"], npz["vel_body"]
            pid = int(npz["platform_id"])
            assert PLATFORMS[pid] == platform, \
                f"{traj_id}: npz platform_id {pid} != index platform {platform!r}"
            quat_mm[frame_off: frame_off + keep] = quat[:keep].astype(np_dtype)

            # The pinned convention (CLAUDE.md finding 3), indexed exactly as
            # repro/build_val_solution.py does, clamp included.
            pi = np.minimum((k + 1) * WIN, n_raw - 1)
            qi = np.minimum(k * WIN + WIN // 2, n_raw - 1)
            p_gt[sl] = pos[pi]
            q_gt[sl] = quat[qi]
            v_gt[sl] = tgt.iloc[sl][["vx", "vy", "vz"]].to_numpy(np.float64)

            # Invariant: the published target IS the per-window mean of vel_body.
            own = vel[:keep].reshape(n_win, WIN, 3).mean(axis=1)
            worst_target_gap = max(worst_target_gap, float(
                np.abs(own - v_gt[sl].astype(np.float32)).max()))

        rows.append({
            "traj_idx": t, "traj_id": traj_id, "split": split,
            "platform": platform, "platform_id": pid,
            "n_windows": n_win, "n_frames_raw": n_raw, "n_frames_kept": keep,
            "frame_offset": frame_off, "window_offset": win_off,
        })
        frame_off += keep
        win_off += n_win

    assert frame_off == n_frames_total and win_off == n_windows_total
    if labelled:
        assert worst_target_gap < 1e-3, \
            (f"{split}: published targets differ from mean(vel_body) by "
             f"{worst_target_gap:.2e} -- the target convention is not what we think")

    imu_mm.flush()
    del imu_mm
    if quat_mm is not None:
        quat_mm.flush()
        del quat_mm
    np.save(out / "window_id.npy", window_id)
    np.save(out / "traj_idx.npy", traj_idx)
    np.save(out / "win_idx.npy", win_idx)
    if labelled:
        np.save(out / "v_gt.npy", v_gt)
        np.save(out / "q_gt.npy", q_gt)
        np.save(out / "p_gt.npy", p_gt)
    traj_df = pd.DataFrame(rows)
    traj_df.to_csv(out / "trajectories.csv", index=False)

    meta = {
        "split": split,
        "cache_version": CACHE_V.name,
        "dtype": dtype,
        "channels": list(CHANNELS_BODY),
        "window_frames": WIN,
        "n_trajectories": len(rows),
        "n_windows": n_windows_total,
        "n_frames": n_frames_total,
        "labelled": labelled,
        "worst_target_vs_mean_velbody": worst_target_gap if labelled else None,
        "position_sampled_at": "window_end",
        "quaternion_sampled_at": "window_mid",
        "quaternion_order": "xyzw",
        "quat_gt_source": "ground_truth_npz" if labelled else None,
        "platforms": {p: int((traj_df.platform == p).sum()) for p in PLATFORMS},
        "source_index_md5": _md5(INDEX / f"{split}_windows.csv"),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "built_by": env_record(),
    }
    write_json(out / "meta.json", meta)

    if verbose:
        mb = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e6
        print(f"[{split}] {len(rows)} trajectories, {n_windows_total} windows, "
              f"{n_frames_total} frames, {mb:.0f} MB, {meta['build_seconds']}s"
              + (f", max |target - mean(vel_body)| = {worst_target_gap:.2e}"
                 if labelled else ""))
    return meta


def build_all(splits=SPLITS, dtype: str = "f16", root: Path | None = None,
              force: bool = False, verbose: bool = True) -> dict[str, dict]:
    return {s: build_split(s, dtype, root, force, verbose) for s in splits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=SPLITS, action="append", default=None)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="f16")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_all(a.split or SPLITS, a.dtype, force=a.force)


if __name__ == "__main__":
    main()
