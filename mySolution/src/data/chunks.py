"""The sample contract: a chunk of consecutive windows. PipelinePlan.md §2.

**The unit the dataset yields is a run of consecutive windows from one
trajectory, not a single window.** Length 1 is a valid configuration and is
mathematically identical to a per-window model (fold the batch and chunk axes
together). Three reasons, in decreasing order of force:

1. The metric-matched ATE20 loss integrates velocities into a path, so it cannot
   be computed from a batch of shuffled, unrelated windows. A window-level
   dataloader forecloses that lever permanently.
2. Trajectory context and trajectory-pooled routing both need consecutive
   windows inside one sample.
3. Nothing is lost -- K windows in, K velocities out, one per window, exactly as
   the submission requires. The chunk changes what *context* is available, never
   what is predicted.

**`chunk_len` is a maximum, not a fixed size.** The measured geometry forces it:
every drone trajectory is at most 60 windows and every human trajectory is at
least 958, so a fixed K = 256 would make every drone chunk 77% padding. Chunks
are grouped into length buckets and padded only within a bucket.

**Masking is mandatory, not optional.** 72.6% of train+val trajectories are
shorter than 64 windows -- all 337 drone, plus 16% of dog. An unmasked mean that
averages in padding zeros is the single most likely silent bug in this pipeline,
and it would corrupt drone specifically, the platform we can least afford to get
wrong. Every key that can be padded carries a documented sentinel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..paths import DT, PLATFORMS, WIN
from ..prep.cache import SplitCache
from ..prep.features import (align_rotation, assemble, is_aligned,
                             n_channels, needs_gravity)
from ..prep.orientation import (degrade_gravity, estimate_gravity,
                                gravity_from_quaternion)
from ..prep.norm import NormStats, identity_stats
from ..prep.segments import SegmentCache
from .augment import Augmenter
from .cond import build_cond
from .timescale import (TimeScaler, out_frames, scaled_truth, time_scale,
                        trajectory_truth)

#: Where the three gravity channels may come from -- `prep.gravity.source`.
#: "truth" is a diagnostic oracle; see ChunkDataset.__init__.
GRAVITY_SOURCES = ("filter", "chunk", "truth")

#: The batch object. Every key is present in every batch regardless of config;
#: keys that do not apply carry a sentinel rather than being omitted. This is
#: what lets a component be switched on by config without touching the loader.
BATCH_KEYS = (
    "x",            # (B,K,C,T) f32   model input                         always present
    "mask",         # (B,K)     bool  True where the window is real       always present
    "traj_idx",     # (B,)      i64   row in the trajectory table         always present
    "win_idx",      # (B,K)     i64   window position in its trajectory   -1 on padding
    "window_id",    # (B,K)     i64   submission row id                   -1 on padding
    "cond",         # (B,4)     f32   conditioning vector fed to the model uniform 0.25
    "v_gt",         # (B,K,3)   f32   target: per-window mean of vel_body zeros on test
    "q_gt",         # (B,K,4)   f32   mid-window quaternion [x,y,z,w]     zeros on test
    "p_gt",         # (B,K,3)   f32   end-window position, world          zeros on test
    "seg_id",       # (B,K)     i64   ATE20 segment index                 -1 if uncontained
    "seg_start",    # (B,S)     i64   ATE20 segment start, chunk-relative -1 on padding
    "seg_end",      # (B,S)     i64   ATE20 segment end, INCLUSIVE        -1 on padding
    "dt",           # (B,K)     f32   always 1.0                          always present
    "platform_id",  # (B,)      i64   true platform, losses/metrics only  -1 on test
    "has_labels",   # (B,)      bool  whether the *_gt keys are real      always present
    "align_R",      # (B,K,3,3) f32   body -> gravity-aligned frame       IDENTITY unless
                    #                                                     input_repr is aligned
)

#: The only three keys the velocity forward pass may read (PipelinePlan.md §2.5).
#: `align_R` is deliberately NOT here: the model never sees it. It is applied to
#: the model's OUTPUT, outside the network, by `models.base.forward_batch`.
FORWARD_KEYS = ("x", "mask", "cond")


@dataclass(frozen=True)
class ChunkSpec:
    """One sample: `length` consecutive windows starting at `start` in trajectory."""
    traj_idx: int
    start: int
    length: int
    bucket: int


def enumerate_chunks(n_windows: int, chunk_len: int, stride: int) -> list[tuple[int, int]]:
    """Cover `[0, n_windows)` with chunks of at most `chunk_len`, stepping `stride`.

    Returns `(start, length)` pairs. Coverage is complete and every window is in
    at least one chunk; with `stride < chunk_len` the chunks overlap and the
    overlap is averaged away later by `post.chain.stitch`. Nothing is padded to a
    length the trajectory cannot fill -- the last chunk is simply shorter.
    """
    if stride > chunk_len:
        raise ValueError(f"chunk_stride {stride} > chunk_len {chunk_len} would leave "
                         f"windows uncovered")
    out, s = [], 0
    while True:
        length = min(chunk_len, n_windows - s)
        out.append((s, length))
        if s + length >= n_windows:
            return out
        s += stride


def enumerate_chunks_snapped(n_windows: int, chunk_len: int,
                             segment_starts) -> list[tuple[int, int]]:
    """Like `enumerate_chunks`, but every chunk starts where an ATE20 segment starts.

    Why bother: a segment only contributes to the ATE20 loss if it fits
    *entirely* inside one chunk, and at an arbitrary offset a segment of length
    L fits in only `K - L + 1` of the `K` possible positions. Snapping the chunk
    start to a segment start is what turns PipelinePlan.md §2.2's coverage table
    from an upper bound into the number you actually get.

    Coverage is still complete: from `pos`, the next chunk begins at the
    furthest segment start that is no further than the end of this chunk, so no
    window is ever skipped. When a single segment is longer than `chunk_len`
    there is no snap point to jump to and the walk falls back to a plain step,
    which simply means that oversized segment is excluded from the ATE20 term.
    """
    starts = np.unique(np.asarray(list(segment_starts) + [0], dtype=np.int64))
    out, pos = [], 0
    while True:
        length = min(chunk_len, n_windows - pos)
        out.append((pos, length))
        if pos + length >= n_windows:
            return out
        reachable = starts[(starts > pos) & (starts <= pos + length)]
        pos = int(reachable[-1]) if len(reachable) else pos + length


def _bucket_for(length: int, buckets: tuple | list) -> int:
    for b in sorted(buckets):
        if length <= b:
            return b
    return int(max(buckets))


def _segment_bounds_of(segments: SegmentCache | None, traj: int,
                       start: int, length: int) -> dict:
    """Chunk-relative inclusive bounds of the whole segments inside one chunk.

    Empty on test, where there are no segments at all, and empty for any chunk
    that happens to contain none -- both are ordinary, not errors. A segment
    straddling a chunk edge is excluded by `contained()`: half a segment cannot
    be aligned against the truth, and the exclusion rate is reported per
    platform by `segment_coverage`.
    """
    if segments is None:
        return {"seg_start": np.zeros(0, np.int64), "seg_end": np.zeros(0, np.int64)}
    _, a, b = segments.contained(traj, start, length)
    return {"seg_start": (a - start).astype(np.int64),
            "seg_end": (b - start).astype(np.int64)}


class ChunkDataset:
    """Yields the §2.5 batch object from a memory-mapped Tier 0 cache."""

    def __init__(self, cache: SplitCache, *, chunk_len: int = 256,
                 chunk_stride: int | None = None, buckets=(64, 128, 256),
                 input_repr: str = "body", extra_scalars=(),
                 traj_subset: list[int] | None = None,
                 segments: SegmentCache | None = None,
                 snap_to_segments: bool = False,
                 cond_source: str = "learned", diagnostic_only: bool = False,
                 posteriors: dict[str, np.ndarray] | None = None,
                 corruption_rate: float = 0.0, cond_seed: int = 0,
                 gravity: np.ndarray | None = None,
                 gravity_source: str = "filter",
                 gravity_cfg: dict | None = None,
                 norm: NormStats | None = None,
                 augment: Augmenter | None = None,
                 timescale: TimeScaler | None = None):
        self.cache = cache
        self.chunk_len = int(chunk_len)
        self.chunk_stride = int(chunk_stride if chunk_stride is not None else chunk_len)
        self.buckets = tuple(sorted(set(buckets) | {self.chunk_len}))
        self.input_repr = input_repr
        self.extra_scalars = tuple(extra_scalars)
        self.n_channels = n_channels(input_repr, self.extra_scalars)
        self.gravity = gravity
        # Where the three gravity channels come from, when the representation
        # wants them (`prep.gravity.source`):
        #
        #   "filter"  the Tier 0 cache, one pass of the complementary filter over
        #             each WHOLE trajectory from its first frame. What every run
        #             up to M7 used.
        #   "chunk"   the same filter, re-run from scratch on the frames of THIS
        #             CHUNK and nothing else. Nothing outside the chunk is read,
        #             so the input becomes a pure function of the sample -- which
        #             is what makes the model self-contained for host-side
        #             re-execution, with no precomputed side-file.
        #
        # The two differ by very little and the difference is measured, not
        # assumed: restarting the filter every 64 windows moves the channel by
        # +0.06 degrees at the macro median, against the channel's own 2.86
        # degrees of error, because the filter converges in about five windows
        # whatever the chunk length. Drone pays exactly zero, its flights being
        # one chunk anyway.
        # .claude/agentTests/2026-09-10_gravity-in-loader/warmup_and_cost.py
        #
        #   "truth"   an ORACLE. The true down-direction, computed per frame from
        #             the ground-truth quaternion -- the same yardstick every
        #             filter angle in this project is measured against. It asks
        #             what a perfect gravity estimate would be worth, and it can
        #             never be submitted: test carries no attitude, and ground
        #             truth at inference is exactly what the rules forbid. So it
        #             is gated like the true-label conditioning sources -- refused
        #             unless run.diagnostic_only is set, which is then stamped
        #             into the result -- and refused outright on unlabelled splits.
        self.gravity_source = str(gravity_source)
        # `prep.gravity` as a whole, for the estimator it names (M19). In "chunk"
        # mode the named estimator runs on the chunk's own frames; an absent
        # estimator is the causal filter, exactly as before.
        self.gravity_cfg = dict(gravity_cfg or {})
        if self.gravity_source not in GRAVITY_SOURCES:
            raise KeyError(f"unknown prep.gravity.source {gravity_source!r}; "
                           f"have {list(GRAVITY_SOURCES)}")
        if needs_gravity(input_repr) and self.gravity_source == "truth":
            if not diagnostic_only:
                raise ValueError(
                    "prep.gravity.source='truth' feeds the model ground-truth attitude. "
                    "It is an oracle, allowed only when run.diagnostic_only is True, and "
                    "the flag is then stamped into the result so the run can never be "
                    "quietly compared against a real one.")
            if cache.quat_gt is None:
                raise ValueError(
                    f"prep.gravity.source='truth' needs ground-truth attitude, and split "
                    f"{cache.split!r} carries none. An oracle model cannot predict test.")
        # M20: the oracle, degraded. `truth_noise_deg` > 0 adds a slowly drifting
        # tilt error with that median to the ground-truth channels, to measure
        # how much of M16's gain survives an imperfect estimate. Absent, the
        # oracle is M16's, exactly. It means nothing without the oracle.
        # M19 lever 3: the gyro-frame ORACLE. The cache was built with each
        # recording's gyro carried into its accelerometer's frame by an axis map
        # fitted against ground truth, so it is gated exactly like the truth
        # source, and exists only as a whole-trajectory cache ("filter").
        if self.gravity_cfg.get("gyro_frame", "raw") == "truth":
            if not diagnostic_only:
                raise ValueError(
                    "prep.gravity.gyro_frame='truth' fits each recording's gyro frame "
                    "against ground truth. It is an oracle, allowed only when "
                    "run.diagnostic_only is True.")
            if self.gravity_source != "filter" or cache.quat_gt is None:
                raise ValueError(
                    "prep.gravity.gyro_frame='truth' needs source='filter' on a split with "
                    f"ground truth; got source {self.gravity_source!r} on {cache.split!r}.")
        # M19 lever 4: a denoised gyro exists only in a whole-trajectory cache --
        # the loader never runs a network.
        if (self.gravity_cfg.get("gyro", "raw") == "denoised"
                and self.gravity_source != "filter"):
            raise ValueError("prep.gravity.gyro='denoised' is built into the gravity cache; "
                             f"it needs source='filter', not {self.gravity_source!r}.")
        self.truth_noise_deg = float(self.gravity_cfg.get("truth_noise_deg", 0.0))
        self.truth_noise_tau_s = float(self.gravity_cfg.get("truth_noise_tau_s", 3.0))
        if self.truth_noise_deg > 0 and self.gravity_source != "truth":
            raise ValueError(
                "prep.gravity.truth_noise_deg degrades the ground-truth oracle and needs "
                f"prep.gravity.source='truth', not {self.gravity_source!r}.")
        self._traj_truth: dict[int, np.ndarray] = {}
        self._dt = 1.0 / float(cache.meta.get("sampling_rate_hz", 200.0))
        # One entry per chunk spec, filled on first use. The estimate depends on
        # nothing but the chunk's own frames and the chunk list is fixed at
        # construction, so computing it once per epoch would be pure waste --
        # the filter is ~21 ms for a 64-window chunk, which is twice the model's
        # own forward pass for the same chunk.
        self._chunk_gravity: dict[int, np.ndarray] = {}
        # Normalisation constants are LOADED, never computed here. Standardising
        # each window by its own statistics would delete the amplitude and
        # frequency differences the embodiment branch depends on -- silently.
        # See prep/norm.py for the full argument.
        self.norm = norm if norm is not None else identity_stats(
            input_repr, self.n_channels)

        if needs_gravity(input_repr) and gravity is None and self.gravity_source == "filter":
            raise ValueError(
                f"input_repr={input_repr!r} needs the per-frame gravity directions. "
                f"Build them once with:\n"
                f"    mySolution/.venv/bin/python -m src.prep.orientation\n"
                f"then pass gravity=load_orientation(split).")
        self.cond_source = cond_source
        self.diagnostic_only = diagnostic_only
        self.posteriors = posteriors
        self.corruption_rate = corruption_rate
        self._rng = np.random.default_rng(cond_seed)
        self.segments = segments
        self.snap_to_segments = bool(snap_to_segments)
        # Augmentation is a TRAINING-ONLY transform and it moves the labels with
        # the input, so a split with no labels has nothing to move and a run
        # that augmented its evaluation would be measuring the wrong thing.
        # `pipeline.build_dataset` passes it to the training dataset alone.
        self.augment = augment
        if augment is not None and not cache.has_labels:
            raise ValueError(
                f"augmentation transforms the target as well as the input, and split "
                f"{cache.split!r} carries no target. Augment the training split only.")
        # Time scaling (data/timescale.py) builds a NEW recording from the chunk's
        # frames and the per-frame truth, so it has the same labelled-only rule, and
        # it refuses the gravity sources it cannot honestly recompute: the oracle's
        # channels are ground truth, and the gyro-frame / denoised estimators live
        # only in whole-trajectory caches built from the ORIGINAL recording.
        self.timescale = timescale
        self._ts_truth: dict[int, dict] = {}
        if timescale is not None:
            if not cache.has_labels:
                raise ValueError(
                    f"time scaling rewrites the target, and split {cache.split!r} carries "
                    f"none. Augment the training split only.")
            if needs_gravity(input_repr) and (
                    self.gravity_source == "truth"
                    or self.gravity_cfg.get("gyro_frame", "raw") != "raw"
                    or self.gravity_cfg.get("gyro", "raw") != "raw"):
                raise ValueError(
                    "time scaling re-estimates the gravity channels on the sped-up IMU; "
                    "it cannot do that for the ground-truth oracle or for an estimator "
                    "that exists only as a whole-trajectory cache.")

        if self.snap_to_segments and segments is None:
            raise ValueError(
                "snap_to_segments=True needs a SegmentCache. Build it once with\n"
                "    mySolution/.venv/bin/python -m src.prep.segments\n"
                "then pass segments=load_segments(split).")

        keep = (range(cache.n_traj) if traj_subset is None else traj_subset)
        self.specs: list[ChunkSpec] = []
        for t in keep:
            n = int(cache.trajectories.n_windows.iloc[t])
            if self.snap_to_segments:
                spans = enumerate_chunks_snapped(
                    n, self.chunk_len, segments.starts_of(int(t)))
            else:
                spans = enumerate_chunks(n, self.chunk_len, self.chunk_stride)
            for start, length in spans:
                self.specs.append(
                    ChunkSpec(int(t), start, length, _bucket_for(length, self.buckets)))

    def __len__(self) -> int:
        return len(self.specs)

    def set_epoch(self, epoch: int) -> None:
        """Tell the augmenter which epoch it is, so the draw changes each pass.

        A no-op when nothing is augmenting. The draw depends on
        `(seed_augment, epoch, chunk index)` and on nothing else, so it is
        reproducible whatever order the batches arrive in.
        """
        if self.augment is not None:
            self.augment.set_epoch(int(epoch))
        if self.timescale is not None:
            self.timescale.set_epoch(int(epoch))

    @property
    def coverage(self) -> dict:
        """How many chunk slots each window occupies -- the stitcher's denominator."""
        counts = np.zeros(self.cache.n_windows, np.int32)
        for s in self.specs:
            sl = self.cache.windows_slice(s.traj_idx, s.start, s.length)
            counts[sl] += 1
        return {"min": int(counts.min()), "max": int(counts.max()),
                "uncovered": int((counts == 0).sum())}

    @property
    def segment_coverage(self) -> dict:
        """Share of ATE20 segments that land whole inside some chunk, per platform.

        This is the honest denominator for the ATE20 loss: everything not
        counted here trains on the velocity term only. Logged into every results
        record as `ate20_segment_coverage`, because a coverage collapse on one
        platform would otherwise look like a modelling result.
        """
        if self.segments is None:
            return {p: None for p in PLATFORMS}
        covered = set()
        for s in self.specs:
            ids, _, _ = self.segments.contained(s.traj_idx, s.start, s.length)
            covered.update(int(i) for i in ids)
        total = {p: 0 for p in PLATFORMS}
        hit = {p: 0 for p in PLATFORMS}
        for t in sorted({s.traj_idx for s in self.specs}):
            platform = str(self.cache.trajectories.platform.iloc[t])
            ids, _, _ = self.segments.bounds_of(t)
            total[platform] += len(ids)
            hit[platform] += sum(1 for i in ids if int(i) in covered)
        return {p: (hit[p] / total[p] if total[p] else None) for p in PLATFORMS}

    def _gravity_block(self, i: int, frames: np.ndarray,
                       lo: int, hi: int) -> np.ndarray | None:
        """The `(L*200, 3)` down-direction for chunk `i`, or None if unused.

        In `"filter"` mode this is a slice of the Tier 0 cache. In `"chunk"` mode
        the filter is run here, on this chunk's frames alone, and memoised -- the
        answer depends on nothing else, so it never needs recomputing. In
        `"truth"` mode it is the ground-truth down-direction of these frames,
        memoised the same way.
        """
        if not needs_gravity(self.input_repr):
            return None
        if self.gravity_source == "filter":
            return (np.asarray(self.gravity[lo:hi], np.float32)
                    if self.gravity is not None else None)
        cached = self._chunk_gravity.get(i)
        if cached is None and self.gravity_source == "truth" and self.truth_noise_deg > 0:
            # The error is drawn for the WHOLE trajectory, once, from a seed tied
            # to its id, then sliced -- so a frame gets the same error in every
            # chunk that contains it, in every epoch and every session.
            s = self.specs[i]
            row = self.cache.trajectories.iloc[s.traj_idx]
            t0 = int(row.frame_offset)
            full = self._traj_truth.get(s.traj_idx)
            if full is None:
                full = degrade_gravity(
                    gravity_from_quaternion(np.asarray(
                        self.cache.quat_gt[t0:t0 + int(row.n_frames_kept)], np.float64)),
                    self.truth_noise_deg, self.truth_noise_tau_s, self._dt,
                    seed_key=str(row.traj_id))
                self._traj_truth[s.traj_idx] = full
            cached = full[lo - t0:hi - t0]
            self._chunk_gravity[i] = cached
        if cached is None:
            if self.gravity_source == "truth":
                # Exactly the reference `prep.orientation.verify` scores the
                # filter against, frame by frame, so "the filter is 6.6 deg off
                # on drone" and "this channel is 0 deg off" use one yardstick.
                cached = gravity_from_quaternion(
                    np.asarray(self.cache.quat_gt[lo:hi], np.float64))
            else:
                # Columns 0:6 are the raw IMU; the estimator reads nothing else,
                # and in particular reads no label and no neighbouring chunk.
                cached = estimate_gravity(np.asarray(frames[:, :6], np.float64),
                                          self._dt, self.gravity_cfg)
            self._chunk_gravity[i] = cached
        return cached

    def __getitem__(self, i: int) -> dict:
        # Time scaling replaces the whole sample -- a new recording, of a new
        # length, with new labels -- so it is decided first. Every chunk it does
        # not draw falls through to the path below, bit for bit unchanged.
        if self.timescale is not None:
            platform = str(self.cache.trajectories.platform.iloc[self.specs[i].traj_idx])
            speed = self.timescale.draw(i, platform)
            # M23: the window grid shifted by 0-199 frames. 0 (the only value when
            # phase_p is unset) leaves every M22 draw exactly as it was.
            phase = self.timescale.draw_phase(i, platform)
            if speed is not None or phase:
                item = self._timescaled_item(i, 1.0 if speed is None else speed, phase)
                if item is not None:
                    return item

        s = self.specs[i]
        c, t, L = self.cache, s.traj_idx, s.length
        row = c.trajectories.iloc[t]
        wsl = c.windows_slice(t, s.start, L)

        frames = np.asarray(c.frames(t, s.start, L))          # (L*200, 6) f16
        lo = int(row.frame_offset) + s.start * WIN
        hi = int(row.frame_offset) + (s.start + L) * WIN
        quat = (np.asarray(c.quat_gt[lo:hi]) if c.quat_gt is not None else None)
        grav = self._gravity_block(i, frames, lo, hi)
        chan = assemble(frames, self.input_repr, quat, self.extra_scalars, grav)

        lab = c.has_labels
        v_gt = (np.asarray(c.v_gt[wsl], np.float32) if lab
                else np.zeros((L, 3), np.float32))
        q_gt = (np.asarray(c.q_gt[wsl], np.float32) if lab
                else np.zeros((L, 4), np.float32))
        p_gt = (np.asarray(c.p_gt[wsl], np.float32) if lab
                else np.zeros((L, 3), np.float32))

        # Augmentation sits here -- after the channels are assembled, before
        # they are normalised -- for two reasons. The normalisation constants
        # are per channel and anisotropic, so rotating a standardised signal
        # would rotate the wrong axes. And the gravity channels already exist by
        # this point, which costs nothing to rotate because the filter that made
        # them is exactly equivariant (data/augment.py). The target moves with
        # the input every time; that is the whole point of PipelinePlan.md §7.
        # The rotation into the gravity-aligned frame, one per window, taken at
        # the window's MIDDLE -- the same instant the scorer samples its
        # quaternion at (CLAUDE.md finding 3), so the target this rotation
        # relates to is sampled where the rotation is exact. The identity for
        # every representation that does not align, which is what keeps every
        # earlier run bit-identical.
        if is_aligned(self.input_repr) and grav is not None:
            mid = np.arange(L) * WIN + WIN // 2
            align_R = align_rotation(np.asarray(grav, np.float32)[mid])
        else:
            align_R = np.broadcast_to(np.eye(3, dtype=np.float32), (L, 3, 3)).copy()

        if self.augment is not None:
            remount = self.augment.draw(
                i, str(row.platform),
                np.asarray(frames[:, :3], np.float64).mean(axis=0))
            chan = remount.channels(chan, needs_gravity(self.input_repr))
            if lab:
                v_gt = remount.vectors(v_gt).astype(np.float32)
                q_gt = remount.quaternions(q_gt)
                p_gt = remount.positions(p_gt)
            # A re-mounted sensor is aligned from its new orientation, so the
            # alignment has to be rebuilt on the rotated gravity -- otherwise the
            # input would be aligned one way and the output rotated back another.
            if is_aligned(self.input_repr):
                mid = np.arange(L) * WIN + WIN // 2
                align_R = align_rotation(
                    np.asarray(chan[:, 6:9], np.float32)[mid]
                    if self.input_repr == "aligned_grav"
                    else remount.vectors(np.asarray(grav, np.float32)[mid]))

        chan = self.norm.apply(chan)
        # frame-major (L*T, C) -> (L, C, T): one reshape and one transpose on a
        # block already in cache. Copy so the tensor owns contiguous memory.
        x = np.ascontiguousarray(
            chan.reshape(L, WIN, self.n_channels).transpose(0, 2, 1))

        return {
            "x": x,
            "window_id": np.asarray(c.window_id[wsl], np.int64),
            "win_idx": np.asarray(c.win_idx[wsl], np.int64),
            "v_gt": v_gt,
            "q_gt": q_gt,
            "p_gt": p_gt,
            # -1 means "no whole ATE20 segment covers this window in this chunk";
            # see prep/segments.py for why a boundary window carries the later id.
            "seg_id": (self.segments.window_seg_id(t, s.start, L)
                       if self.segments is not None else np.full(L, -1, np.int64)),
            # The bounds the ATE20 loss integrates between, chunk-relative and
            # END-INCLUSIVE. They are carried in addition to `seg_id` rather than
            # derived from it because one integer per window cannot say that a
            # boundary window belongs to two segments -- and the scorer's
            # segments do share their boundary window, so it must contribute to
            # both errors (prep/segments.py, CLAUDE.md finding 9).
            **_segment_bounds_of(self.segments, t, s.start, L),
            "align_R": align_R,
            "traj_idx": t,
            "traj_id": str(row.traj_id),
            "platform_id": int(row.platform_id),
            "has_labels": bool(lab),
        }

    def _timescaled_item(self, i: int, speed: float, phase: int = 0) -> dict | None:
        """Chunk `i` played back `speed` times faster, as a complete sample.

        The chunk's own start is kept; enough original windows are read after it
        to fill the chunk at the new speed, capped by the trajectory's end -- so a
        drone flight, which is one chunk, simply comes out shorter (a 60 s flight
        at 1.25x is 48 windows). The gravity channels are re-estimated on the new
        IMU by the run's own estimator, exactly as `"chunk"` mode would on a real
        recording. Synthetic windows are no submission row, so `window_id` and
        `win_idx` are -1; they carry no ATE20 segment either (the term is off in
        every submitted config, and a new time base would need new segments).
        Returns None when not even one whole window fits.

        `phase` (M23) starts the new window grid that many frames later. At
        `speed == 1` nothing is transformed at all: the sample is the recording's
        own frames, cut into windows at a different place, with labels that are
        the window means of the recorded per-frame truth, and gravity channels
        that are the run's cached estimate at those very frames -- exactly what
        the model would read if the recording had started `phase` frames later.
        """
        spec = self.specs[i]
        c, t = self.cache, spec.traj_idx
        row = c.trajectories.iloc[t]
        n_in_win = min(int(row.n_windows) - spec.start, int(np.ceil(speed * spec.length)) + 1)
        n_in = n_in_win * WIN - phase
        n_win = min(spec.length, out_frames(n_in, speed) // WIN)
        if n_win < 1:
            return None
        n_out = n_win * WIN

        truth = self._ts_truth.get(t)
        if truth is None:
            truth = trajectory_truth(c, t, self.timescale.split_box_seconds)
            self._ts_truth[t] = truth
        f0 = spec.start * WIN + phase
        frames = np.asarray(c.frames(t, spec.start, n_in_win), np.float64)[phase:]
        if speed == 1.0:
            imu = frames[:n_out]
            acc = imu[:, 0:3]
        else:
            acc, gyr, _ = time_scale(frames[:, 0:3], frames[:, 3:6],
                                     truth["down"][f0:f0 + n_in], speed, n_out,
                                     self.timescale.fc_hz)
            imu = np.concatenate([acc, gyr], axis=1)
        lab = scaled_truth(truth, f0, n_in, speed, n_out)
        v_gt, q_gt, p_gt = lab["v_gt"], lab["q_gt"], lab["p_gt"]

        if not needs_gravity(self.input_repr):
            grav = None
        elif speed == 1.0 and self.gravity_source == "filter" and self.gravity is not None:
            # Same frames, so the cached estimate is still the right one -- and it
            # is what the model reads at test time, trajectory-wide, warm-up included.
            lo = int(row.frame_offset) + f0
            grav = np.asarray(self.gravity[lo:lo + n_out], np.float32)
        else:
            grav = estimate_gravity(imu, self._dt, self.gravity_cfg)
        chan = assemble(imu, self.input_repr, None, self.extra_scalars, grav)

        # The same tail as the unscaled path: re-mounting on top when both ops are
        # on, the alignment rotation at each window's middle, then normalisation.
        if is_aligned(self.input_repr) and grav is not None:
            mid = np.arange(n_win) * WIN + WIN // 2
            align_R = align_rotation(np.asarray(grav, np.float32)[mid])
        else:
            align_R = np.broadcast_to(np.eye(3, dtype=np.float32), (n_win, 3, 3)).copy()
        if self.augment is not None:
            remount = self.augment.draw(i, str(row.platform), acc.mean(axis=0))
            chan = remount.channels(chan, needs_gravity(self.input_repr))
            v_gt = remount.vectors(v_gt).astype(np.float32)
            q_gt = remount.quaternions(q_gt)
            p_gt = remount.positions(p_gt)
            if is_aligned(self.input_repr):
                mid = np.arange(n_win) * WIN + WIN // 2
                align_R = align_rotation(
                    np.asarray(chan[:, 6:9], np.float32)[mid]
                    if self.input_repr == "aligned_grav"
                    else remount.vectors(np.asarray(grav, np.float32)[mid]))

        chan = self.norm.apply(chan)
        x = np.ascontiguousarray(chan.reshape(n_win, WIN, self.n_channels).transpose(0, 2, 1))
        return {
            "x": x,
            "window_id": np.full(n_win, -1, np.int64),
            "win_idx": np.full(n_win, -1, np.int64),
            "v_gt": v_gt,
            "q_gt": q_gt,
            "p_gt": p_gt,
            "seg_id": np.full(n_win, -1, np.int64),
            "seg_start": np.zeros(0, np.int64),
            "seg_end": np.zeros(0, np.int64),
            "align_R": align_R,
            "traj_idx": t,
            "traj_id": str(row.traj_id),
            "platform_id": int(row.platform_id),
            "has_labels": True,
            "speed": float(speed),
            "phase": int(phase),
        }

    # ------------------------------------------------------------------ collate

    def cond_for(self, platform_ids: np.ndarray, traj_ids: list[str]) -> np.ndarray:
        return build_cond(
            self.cond_source, platform_ids, diagnostic_only=self.diagnostic_only,
            posteriors=self.posteriors, traj_ids=traj_ids,
            corruption_rate=self.corruption_rate, rng=self._rng)

    def collate(self, samples: list[dict]) -> dict:
        return collate(samples, cond_fn=self.cond_for)

    def batch(self, indices) -> dict:
        return self.collate([self[i] for i in indices])


class ConcatChunkDataset:
    """Several ChunkDatasets trained as one -- train plus val, for a final model.

    Index `i` runs through the parts in order, and each chunk is read by the part
    that owns it, so it keeps its own cache, gravity file and trajectory numbering.
    Nothing downstream of the batch reads `traj_idx`, so two parts reusing the same
    small integers for different trajectories is harmless.
    """

    def __init__(self, parts: list[ChunkDataset]):
        self.parts = list(parts)
        first = self.parts[0]
        for p in self.parts[1:]:
            if (p.n_channels, p.chunk_len, p.input_repr) != (
                    first.n_channels, first.chunk_len, first.input_repr):
                raise ValueError("every part must read the same channels at the same chunk length")
        self.n_channels, self.chunk_len = first.n_channels, first.chunk_len
        self.input_repr, self.augment = first.input_repr, first.augment
        self.timescale = first.timescale
        self.specs = [s for p in self.parts for s in p.specs]
        self._starts = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, i: int) -> dict:
        k = int(np.searchsorted(self._starts, i, side="right")) - 1
        return self.parts[k][i - int(self._starts[k])]

    def set_epoch(self, epoch: int) -> None:
        for p in self.parts:
            p.set_epoch(epoch)

    def chunk_platform_ids(self) -> np.ndarray:
        """Platform of every chunk, read per part from that part's own cache."""
        return np.concatenate([
            p.cache.trajectories.platform_id.to_numpy()[[s.traj_idx for s in p.specs]]
            for p in self.parts]).astype(np.int64)

    def batch(self, indices) -> dict:
        return self.parts[0].collate([self[i] for i in indices])


def collate(samples: list[dict], cond_fn=None) -> dict:
    """Pad a list of chunks to the batch's longest and build the §2.5 batch object.

    Padding is to the longest chunk *in this batch*, not to the bucket ceiling --
    equivalent, and cheaper.
    """
    b = len(samples)
    k = max(s["x"].shape[0] for s in samples)
    c, t = samples[0]["x"].shape[1], samples[0]["x"].shape[2]
    # Segment slots are padded like everything else. A batch whose chunks all
    # contain no whole segment gives S = 0, which the ATE20 term reads as "no
    # trajectory signal here" rather than as an error.
    n_seg = max((len(s["seg_start"]) for s in samples), default=0)

    x = np.zeros((b, k, c, t), np.float32)
    mask = np.zeros((b, k), bool)
    win_idx = np.full((b, k), -1, np.int64)
    window_id = np.full((b, k), -1, np.int64)
    v_gt = np.zeros((b, k, 3), np.float32)
    align_R = np.broadcast_to(np.eye(3, dtype=np.float32), (b, k, 3, 3)).copy()
    q_gt = np.zeros((b, k, 4), np.float32)
    p_gt = np.zeros((b, k, 3), np.float32)
    seg_id = np.full((b, k), -1, np.int64)
    seg_start = np.full((b, n_seg), -1, np.int64)
    seg_end = np.full((b, n_seg), -1, np.int64)
    traj_idx = np.zeros(b, np.int64)
    platform_id = np.full(b, -1, np.int64)
    has_labels = np.zeros(b, bool)
    traj_ids: list[str] = []

    for i, s in enumerate(samples):
        n = s["x"].shape[0]
        x[i, :n] = s["x"]
        mask[i, :n] = True
        win_idx[i, :n] = s["win_idx"]
        window_id[i, :n] = s["window_id"]
        v_gt[i, :n] = s["v_gt"]
        align_R[i, :n] = s["align_R"]
        q_gt[i, :n] = s["q_gt"]
        p_gt[i, :n] = s["p_gt"]
        seg_id[i, :n] = s["seg_id"]
        seg_start[i, :len(s["seg_start"])] = s["seg_start"]
        seg_end[i, :len(s["seg_end"])] = s["seg_end"]
        traj_idx[i] = s["traj_idx"]
        platform_id[i] = s["platform_id"]
        has_labels[i] = s["has_labels"]
        traj_ids.append(s["traj_id"])

    cond = (cond_fn(platform_id, traj_ids) if cond_fn is not None
            else np.full((b, 4), 0.25, np.float32))

    batch = {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "traj_idx": torch.from_numpy(traj_idx),
        "win_idx": torch.from_numpy(win_idx),
        "window_id": torch.from_numpy(window_id),
        "cond": torch.from_numpy(np.ascontiguousarray(cond, np.float32)),
        "v_gt": torch.from_numpy(v_gt),
        "align_R": torch.from_numpy(align_R),
        "q_gt": torch.from_numpy(q_gt),
        "p_gt": torch.from_numpy(p_gt),
        "seg_id": torch.from_numpy(seg_id),
        "seg_start": torch.from_numpy(seg_start),
        "seg_end": torch.from_numpy(seg_end),
        "dt": torch.full((b, k), DT, dtype=torch.float32),
        "platform_id": torch.from_numpy(platform_id),
        "has_labels": torch.from_numpy(has_labels),
    }
    assert set(batch) == set(BATCH_KEYS), \
        f"batch keys drifted from the §2.5 contract: {set(batch) ^ set(BATCH_KEYS)}"
    batch["traj_id"] = traj_ids          # python strings, not a tensor; not a batch key
    return batch
