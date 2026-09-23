"""Batch composition. PipelinePlan.md §4.5, models.md §8.

**This is the part that most often goes wrong, and it goes wrong silently.**

The score treats the four platforms as equally important: it averages the error
within each platform first, then averages those four numbers. Training does no
such thing unless it is made to. And the data is skewed in two *opposite*
directions at once, so neither obvious sampling scheme is close to right:

                        car     dog     drone    human    (what the metric wants)
    uniform by window   28.7%   18.2%   17.4%    35.7%           25% each
    uniform by traj.    11.1%    9.1%   73.2%     6.6%           25% each

Sampling windows uniformly starves drone and dog, because drone flights are short
-- 12 to 60 seconds against human recordings that run 16 minutes. Sampling whole
trajectories uniformly does the reverse and produces, in effect, a drone-only
model, because there are 289 drone recordings against 26 human ones.

`platform_balanced` fixes this the direct way: build every batch out of an equal
number of chunks from each platform. With four chunks per platform the batch is
16, which divides exactly.

**A subtler distortion remains inside each platform**, and it is deliberately
left for later. The metric weights every trajectory equally regardless of its
length, so a window's true weight is `1 / (4 * trajectories_in_platform *
windows_in_its_trajectory)`. Drawing chunks uniformly within a platform still
over-weights the long recordings. Correcting that is the `within_platform_
weighting` option and an M11 experiment, not a day-one requirement.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MODES = ("sequential", "window_uniform", "trajectory_uniform", "platform_balanced")


def build_batches(dataset, batch_size: int, mode: str = "sequential",
                  bucketed: bool = True, seed: int = 0,
                  drop_last: bool = False) -> list[list[int]]:
    """Return a list of index lists, one per batch.

    `sequential` is the inference order: chunks in dataset order, grouped by
    length bucket so a batch pads as little as possible. Every other mode is a
    training order and shuffles.
    """
    if mode == "sequential":
        return _sequential(dataset, batch_size, bucketed)
    if mode == "platform_balanced":
        return _platform_balanced(dataset, batch_size, seed)
    if mode in ("window_uniform", "trajectory_uniform"):
        raise NotImplementedError(
            f"sampler.mode={mode!r} is an M11 ablation (models.md §10). It exists to "
            f"be *compared against* platform_balanced, and the comparison has not "
            f"been reached yet.")
    raise KeyError(f"unknown sampler mode {mode!r}; have {list(MODES)}")


def _sequential(dataset, batch_size: int, bucketed: bool) -> list[list[int]]:
    """Dataset order, optionally grouped by length so batches pad less."""
    if not bucketed:
        idx = list(range(len(dataset)))
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]

    by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, spec in enumerate(dataset.specs):
        by_bucket[spec.bucket].append(i)
    batches: list[list[int]] = []
    for bucket in sorted(by_bucket):
        idx = by_bucket[bucket]
        batches += [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    return batches


def chunk_platform_ids(dataset) -> np.ndarray:
    """Which platform each chunk in the dataset belongs to, as `(n_chunks,)` int.

    Read from the cache's trajectory table rather than from the batch, because
    batches do not exist yet at the point batches are being planned.
    """
    if hasattr(dataset, "chunk_platform_ids"):        # several caches, one dataset
        return dataset.chunk_platform_ids()
    platform_of_traj = dataset.cache.trajectories.platform_id.to_numpy()
    return np.asarray([platform_of_traj[s.traj_idx] for s in dataset.specs], np.int64)


def _platform_balanced(dataset, batch_size: int, seed: int) -> list[list[int]]:
    """Equal chunks per platform in every batch.

    Each platform's chunks are shuffled into its own queue and then dealt out
    round-robin. The platforms have wildly different queue lengths, so the short
    queues are refilled by reshuffling when they run dry: over one epoch a drone
    chunk is therefore seen several times and a human chunk less than once. That
    is the intended behaviour -- "an epoch" stops meaning "one pass over the
    data" and starts meaning "a fixed number of balanced steps", which is what
    matching the metric requires.

    Epoch length is set by the *largest* platform, so no platform's data is
    systematically left unseen across successive epochs.
    """
    platform_ids = chunk_platform_ids(dataset)
    present = sorted(set(int(p) for p in platform_ids))
    if not present:
        return []
    per_platform = batch_size // len(present)
    if per_platform < 1:
        raise ValueError(
            f"batch_size {batch_size} cannot hold one chunk from each of "
            f"{len(present)} platforms")
    if batch_size % len(present):
        raise ValueError(
            f"batch_size {batch_size} does not divide evenly among "
            f"{len(present)} platforms; the batch would be quietly imbalanced")

    rng = np.random.default_rng(seed)
    pools = {p: np.flatnonzero(platform_ids == p) for p in present}
    queues = {p: list(rng.permutation(pools[p])) for p in present}

    n_batches = max(len(pools[p]) for p in present) // per_platform
    batches: list[list[int]] = []
    for _ in range(n_batches):
        batch: list[int] = []
        for p in present:
            for _ in range(per_platform):
                if not queues[p]:                    # this platform ran out; reshuffle
                    queues[p] = list(rng.permutation(pools[p]))
                batch.append(int(queues[p].pop()))
        rng.shuffle(batch)      # so position in the batch carries no platform signal
        batches.append(batch)
    return batches


def platform_share(dataset, batches: list[list[int]]) -> dict[int, float]:
    """What fraction of the sampled chunks each platform actually got.

    A diagnostic worth logging: the whole point of the balanced sampler is a
    number close to 0.25 for each platform, and this is how that claim is
    checked rather than assumed.
    """
    platform_ids = chunk_platform_ids(dataset)
    counts: dict[int, int] = defaultdict(int)
    total = 0
    for batch in batches:
        for i in batch:
            counts[int(platform_ids[i])] += 1
            total += 1
    return {p: counts[p] / total for p in sorted(counts)} if total else {}
