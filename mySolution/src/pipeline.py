"""The runner: cache -> dataset -> model -> post -> submission -> score -> run log.

    mySolution/.venv/bin/python -m src.pipeline                     # M0 on val
    mySolution/.venv/bin/python -m src.pipeline --split test
    mySolution/.venv/bin/python -m src.pipeline --set data.chunk_len=64

This is the only path that produces a submission, and it is where the two hard
rules of PipelinePlan.md §2.5 are *enforced* rather than merely written down:

1. The velocity forward pass is called through `models.forward_batch`, which
   hands the model a three-key subset of the batch. A model cannot reach
   `platform_id`, `q_gt`, `p_gt` or `v_gt` even by accident.
2. A submission-producing run must derive `cond` from something other than the
   true label. `data.cond.build_cond` refuses the gated sources unless
   `run.diagnostic_only` is set, and that flag is stamped into the results
   record so a ceiling experiment can never be quietly compared against a real
   one. This is precisely the defect that makes the published 0.637 baseline
   unreproducible by any participant; it is not repeated here.

Every run writes `runs/<YYYYMMDD>-<name>-<confighash8>/` with config.json,
env.json, metrics.json and predictions/, and appends one line to
`runs/results.jsonl`. **The report's ablation tables are generated from that
file by a script, never assembled by hand**, which is why its schema is fixed
here at M0 and already carries fields nothing uses yet.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import config as config_mod
from .data.augment import build_augmenter
from .data.chunks import ChunkDataset, ConcatChunkDataset
from .data.timescale import build_timescaler
from .data.sampler import build_batches
from .evaluation.score import format_report, score_submission
from .losses import build_loss, gravity_angles_deg
from .metric_constants import (VAL_BASELINE_FORCED, VAL_BASELINE_ROUTED,
                               VAL_FLOOR, VAL_ZEROS)
from .models import build_model, forward_batch
from .paths import PLATFORMS, RESULTS_JSONL
from .post.chain import PredictionAccumulator, run_chain
from .post.submission import write_submission
from .prep import norm as norm_mod
from .prep.cache import cache_exists, load_cache
from .prep.features import n_channels, needs_gravity
from .prep.orientation import estimator_key, load_orientation
from .losses import term_enabled
from .prep.segments import (LABELLED_SPLITS, load_segments,
                            segments_available)
from .prep.splits import load_splits, record_opening
from .utils import (Timer, append_jsonl, config_hash_appears, env_record,
                    run_dir, seed_everything, write_json)


def segments_possible(split: str) -> bool:
    """Can this split have ATE20 segments at all? Test never can -- it has no `pos`."""
    return split in LABELLED_SPLITS


def build_dataset(cfg, split: str, traj_subset: list[int] | None = None,
                  augment: bool = False) -> ChunkDataset:
    """Assemble the §2.5 dataset for one split, honouring every data config key.

    Three optional Tier 0 caches feed in here, and each is loaded only when the
    config actually asks for it: the ATE20 segment bounds, the per-frame gravity
    directions, and the fixed input normalisation constants.

    `augment` is False everywhere but the training set. It is an argument rather
    than a config read so that the one place augmentation can switch on is
    visible in the call, not buried in a dictionary: evaluating on transformed
    data would silently measure a different problem from the one the leaderboard
    scores.
    """
    if not cache_exists(split):
        raise FileNotFoundError(
            f"no Tier 0 cache for {split!r}. Build it first:\n"
            f"    mySolution/.venv/bin/python -m src.prep.windows")
    cache = load_cache(split)
    d, c = cfg["data"], cfg["cond"]

    # Segments exist for train and val only -- test NPZ files carry no position,
    # so there is no path to cut into 20 m pieces. Loading them is what fills the
    # batch's seg_id key and the run record's ate20_segment_coverage field.
    want_segments = (cfg["prep"]["segments"]["enabled"] or d["snap_to_segments"]
                     or term_enabled(cfg, "ate20"))
    segments = (load_segments(split)
                if want_segments and segments_available(split) else None)
    # Test has no segments and never will, so a config that wants them is still
    # a valid config to *predict* with -- the ATE20 term is a training signal
    # only, and host-side re-execution runs exactly this path on raw test NPZ.
    # Demanding a cache that cannot exist would make a trained model unusable.
    if want_segments and segments is None and segments_possible(split):
        raise FileNotFoundError(
            f"this config needs ATE20 segments but split {split!r} has no segment "
            f"cache. Build it with:  mySolution/.venv/bin/python -m src.prep.segments")

    # `prep.gravity.source` decides where the three gravity channels come from.
    # "filter" reads the Tier 0 cache, one pass over each whole trajectory.
    # "chunk" re-runs the same filter inside the loader on each chunk's own
    # frames, so the input is a pure function of the sample and the model needs
    # no precomputed side-file to reproduce its predictions from raw test NPZ.
    # Measured difference: +0.06 degrees at chunk_len 64, against the channel's
    # own 2.86 -- see data/chunks.py. "truth" is a diagnostic oracle that feeds
    # the ground-truth down-direction instead; ChunkDataset refuses it without
    # run.diagnostic_only, and on any split with no attitude (test).
    # `prep.gravity.estimator` (M19) says WHICH estimator made the channels; an
    # absent key is the causal filter, read from its original cache file, so no
    # earlier run changes. Any other estimator reads its own keyed file.
    gravity_cfg = cfg["prep"].get("gravity", {})
    gravity_source = gravity_cfg.get("source", "filter")
    gravity = (load_orientation(split, key=estimator_key(gravity_cfg))
               if needs_gravity(d["input_repr"]) and gravity_source == "filter"
               else None)
    norm = _load_norm(d)

    return ChunkDataset(
        cache,
        chunk_len=d["chunk_len"], chunk_stride=d["chunk_stride"],
        buckets=d["buckets"], input_repr=d["input_repr"],
        extra_scalars=d["extra_scalars"], segments=segments,
        # Snapping chunk starts to segment starts is meaningless without
        # segments, which is the test split's permanent condition.
        snap_to_segments=d["snap_to_segments"] and segments is not None,
        traj_subset=traj_subset,
        cond_source=c["source"], diagnostic_only=cfg["run"]["diagnostic_only"],
        corruption_rate=c["corruption_rate"], cond_seed=cfg["run"]["seed_data"],
        gravity=gravity, gravity_source=gravity_source, gravity_cfg=gravity_cfg,
        norm=norm, augment=build_augmenter(cfg) if augment else None,
        timescale=build_timescaler(cfg) if augment else None)


def _load_norm(data_cfg):
    """The fixed input normalisation constants named by `data.normalization`.

    `dataset_fixed` loads numbers measured once over train. `none` leaves the
    input alone. There is deliberately no third option: standardising each
    window by its own statistics would delete the amplitude and frequency cues
    the embodiment branch depends on, silently (prep/norm.py).
    """
    mode = data_cfg.get("normalization", "none")
    channels = n_channels(data_cfg["input_repr"], data_cfg["extra_scalars"])
    if mode == "none":
        return norm_mod.identity_stats(data_cfg["input_repr"], channels)
    if mode == "dataset_fixed":
        return norm_mod.load(data_cfg["input_repr"])
    raise KeyError(
        f"unknown data.normalization {mode!r}; have 'none' and 'dataset_fixed'. "
        f"Per-window standardisation is not offered on purpose -- see prep/norm.py.")


def predict_dataset(model, dataset, cfg, window_ids=None, verbose: bool = True):
    """Run `model` over every chunk in `dataset` and stitch the overlaps together.

    `window_ids` is the ascending id list the stitcher reconciles against. It
    defaults to the whole split; pass a subset when predicting only part of one,
    as scoring the sealed holdout does.
    """
    cache = dataset.cache
    device = torch.device(cfg["run"]["device"])
    model = model.to(device).eval()
    if window_ids is None:
        window_ids = cache.window_id

    batches = build_batches(dataset, cfg["data"]["batch_size"], "sequential")
    acc = PredictionAccumulator(
        window_ids, cfg["post"].get("stitch", {}).get("weighting", "uniform"))
    loss_fn = build_loss(cfg)
    loss_sum, loss_n = 0.0, 0
    # M15: per-window gravity angles, bucketed by platform, when the model has a
    # gravity head and the split has the quaternions to score it against.
    gravity_angles: dict[int, list[np.ndarray]] | None = None

    with torch.no_grad():
        for bi, idx in enumerate(batches):
            batch = dataset.batch(idx)
            gpu = {k: (v.to(device) if torch.is_tensor(v) else v)
                   for k, v in batch.items()}
            out = forward_batch(model, gpu)          # sees x, mask, cond only
            if cache.has_labels:
                _, parts = loss_fn(out, gpu)
                n = int(gpu["mask"].sum())
                loss_sum += parts["score_proxy"] * n
                loss_n += n
                if "gravity" in out:
                    angles, valid = gravity_angles_deg(out["gravity"], gpu["q_gt"],
                                                       gpu["mask"])
                    pid = gpu["platform_id"][:, None].expand_as(valid)[valid].cpu().numpy()
                    angles = angles.cpu().numpy()
                    gravity_angles = gravity_angles if gravity_angles is not None else {}
                    for p_i in np.unique(pid):
                        gravity_angles.setdefault(int(p_i), []).append(angles[pid == p_i])
            acc.add(batch["window_id"].numpy(),
                    out["velocity"].detach().cpu().numpy(),
                    batch["mask"].numpy())
            if verbose and (bi + 1) % 100 == 0:
                print(f"  batch {bi + 1}/{len(batches)}")

    table = run_chain(acc.table(), cfg)
    stats = {
        "n_chunks": len(dataset), "n_batches": len(batches),
        "coverage": dataset.coverage, "n_parameters": model.n_parameters(),
        "segment_coverage": dataset.segment_coverage,
        "train_loss_proxy": (loss_sum / loss_n) if loss_n else None,
        "gravity_angle_deg": _gravity_summary(gravity_angles),
    }
    return table, stats


def _gravity_summary(angles: dict[int, list] | None) -> dict | None:
    """Median gravity angle per platform, and their plain mean: the macro-median.

    The macro-median is what M15's first kill criterion reads -- below 10
    degrees, or the head has not learned the quantity at all. It is averaged
    over platforms equally for the same reason the score is, and it is a
    *median* within each platform because a handful of violent manoeuvres
    should not decide whether the head learned tilt. `None` when there is
    nothing to summarise: no gravity head, or a split with no quaternions.
    """
    if not angles:
        return None
    per = {PLATFORMS[p]: float(np.median(np.concatenate(a)))
           for p, a in sorted(angles.items()) if 0 <= p < len(PLATFORMS)}
    return {"per_platform": per,
            "macro_median": float(np.mean(list(per.values()))) if per else None}


def predict_split(cfg, split: str, verbose: bool = True, model=None):
    """Run the model over a whole split and return the stitched prediction table.

    With `model=None` a fresh, untrained model is built -- which is exactly what
    M0's all-zeros check wants. A trained model is passed in by `run()`.
    """
    dataset = build_dataset(cfg, split)
    if model is None:
        model = build_model(cfg, in_channels=dataset.n_channels)
    table, stats = predict_dataset(model, dataset, cfg, verbose=verbose)
    return table, dataset.cache, stats


def window_ids_of(dataset) -> np.ndarray:
    """The ascending window ids covered by a dataset's chunks."""
    ids: list[np.ndarray] = []
    for spec in dataset.specs:
        sl = dataset.cache.windows_slice(spec.traj_idx, spec.start, spec.length)
        ids.append(np.asarray(dataset.cache.window_id[sl]))
    return np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)


def build_training_dataset(cfg, verbose: bool = True) -> ChunkDataset:
    """The chunks a run is allowed to fit on: train, minus the sealed holdout.

    The seal is 61 trajectories drawn once with seed 42 and committed to
    `configs/splits_v1.json`. Nothing may train on them, ever -- they are how we
    find out at the end whether three weeks of tuning against val was real.
    Excluding them here, in the one function that builds training data, is what
    makes that a property of the code rather than a promise.
    """
    cache = load_cache(cfg["data"]["train_split"])
    keep = list(range(cache.n_traj))
    if cfg["data"].get("exclude_seal", True):
        sealed = set(load_splits().seal_rows(cache))
        keep = [t for t in keep if t not in sealed]
        if verbose:
            print(f"training on {len(keep)} of {cache.n_traj} train trajectories "
                  f"({len(sealed)} sealed and excluded)")
    dataset = build_dataset(cfg, cfg["data"]["train_split"], traj_subset=keep,
                            augment=True)
    # `data.train_extra`: whole labelled splits trained on as well -- val, for a
    # final model. Absent for every run before 2026-09-18, so no hash moves.
    extra = list(cfg["data"].get("train_extra") or [])
    if extra:
        parts = [dataset] + [build_dataset(cfg, s, augment=True) for s in extra]
        dataset = ConcatChunkDataset(parts)
        if verbose:
            print(f"also training on every trajectory of {extra}: "
                  + ", ".join(f"{s} {p.cache.n_traj} trajectories / {len(p)} chunks"
                              for s, p in zip(extra, parts[1:])))
    if verbose and dataset.augment is not None:
        a = dataset.augment
        print(f"augmenting: yaw(deg) "
              + " ".join(f"{k}={v:g}" for k, v in sorted(a.yaw_deg.items()))
              + f"  tilt={a.tilt_deg:g}deg  "
              + "mirror " + " ".join(f"{k}={v:g}" for k, v in sorted(a.mirror_p.items()))
              + f"  seed={a.seed}")
    if verbose and getattr(dataset, "timescale", None) is not None:
        ts = dataset.timescale
        print("time scaling: " + "  ".join(
            f"{q} p={ts.p[q]:g} s={ts.s_range[q][0]:g}..{ts.s_range[q][1]:g}"
            for q in sorted(ts.p) if ts.p[q] > 0)
            + f"  vibration kept above {ts.fc_hz:g} Hz  seed={ts.seed}")
        if any(v > 0 for v in ts.phase_p.values()):
            print("window phase (M23): " + "  ".join(
                f"{q} p={v:g}" for q, v in sorted(ts.phase_p.items()) if v > 0)
                + "  -- window grid shifted 1-199 frames, real frames and real labels")
    return dataset


def score_seal(cfg, model, reason: str, verbose: bool = True) -> dict:
    """Predict and score the sealed holdout, and write the opening to the audit log.

    **Read this number as a change, never as a level.** The seed-42 draw is
    measurably about one standard error harder than the train population it came
    from, so its absolute score means little; comparing seal to seal across model
    versions cancels that offset out. And its noise floor is about 0.0155, so a
    val-versus-seal gap under roughly 0.03 is not evidence of anything at all.
    """
    split = cfg["data"]["train_split"]
    cache = load_cache(split)
    rows = load_splits().seal_rows(cache)
    dataset = build_dataset(cfg, split, traj_subset=rows)
    ids = window_ids_of(dataset)
    table, _ = predict_dataset(model, dataset, cfg, window_ids=ids, verbose=False)

    traj_ids = [str(cache.trajectories.traj_id.iloc[t]) for t in rows]
    result = score_submission(table, split, traj_ids=traj_ids)
    record_opening(reason, result["score"], n_trajectories=len(rows),
                   config_hash=cfg.hash, run_name=cfg["run"]["name"])
    if verbose:
        print()
        print(format_report(result, f"SEALED HOLDOUT ({len(rows)} trajectories)"))
    return result


def run(reference: str = "m0_zero", overrides: dict | None = None,
        split: str | None = None, rerun: bool = False,
        verbose: bool = True) -> dict:
    # The split is part of the science, so it goes into the config -- and
    # therefore into the hash -- rather than staying a loose runtime argument.
    overrides = dict(overrides or {})
    if split is not None:
        overrides.setdefault("data", {})["splits"] = [split]
        overrides.setdefault("eval", {})["splits"] = [split]
    cfg = config_mod.resolve(reference, overrides)
    split = cfg["eval"]["splits"][0]

    if config_hash_appears(cfg.hash) and not rerun:
        raise SystemExit(
            f"config hash {cfg.hash} is already in {RESULTS_JSONL}. The science of "
            f"this run has been done. Pass --rerun to do it again anyway.")

    seed_everything(cfg["run"]["seed_weights"])
    out_dir = run_dir(cfg["run"]["name"], cfg.hash)
    model, history = None, None

    with Timer() as t:
        if int(cfg["optim"]["epochs"]) > 0:
            model, history = _fit(cfg, out_dir, verbose)
        table, cache, stats = predict_split(cfg, split, verbose, model=model)
        sub_path = write_submission(
            table, out_dir / "predictions" / f"submission_{split}.csv",
            cache.window_id, split)
        # A split the model trained on is predicted but never scored: the number
        # would be a training error, and results.jsonl would file it as val.
        trained_on = split in (cfg["data"].get("train_extra") or [])
        result = (score_submission(sub_path, split)
                  if cache.has_labels and not trained_on else None)
        seal = (score_seal(cfg, model, f"{cfg['run']['milestone']}: "
                           f"{cfg['run']['name']}", verbose)
                if cfg["eval"].get("seal") and model is not None else None)

    record = _results_record(cfg, split, result, stats, t.seconds, out_dir,
                             history=history, seal=seal)
    write_json(out_dir / "config.json",
               {"config": dict(cfg), **cfg.provenance()})
    write_json(out_dir / "env.json", env_record())
    write_json(out_dir / "metrics.json", record["metrics"] or {})
    append_jsonl(RESULTS_JSONL, record)

    if verbose:
        print(f"\nrun dir     {out_dir}")
        print(f"submission  {sub_path}  ({len(table)} rows)")
        print(f"config      {cfg.reference_name} v{cfg.reference_version} "
              f"#{cfg.hash}  diff={cfg.diff or '{}'}")
        if result:
            print()
            print(format_report(result, f"{cfg['run']['name']} on {split}"))
            print(_bar_context(result["score"]))
        elif trained_on:
            print(f"\n{split} is in the training data (data.train_extra), so it is not scored.")
        else:
            print(f"\n{split} carries no labels, so it cannot be scored locally. "
                  f"ATE20 has no segments on test (no `pos` in the NPZ files).")
    return record


def selection_set(cfg) -> str:
    """Where `_fit` scores each epoch: "val" (the default), "seal" or "none".

    Refuses the two combinations that would choose weights on data the model
    trained on, because nothing would crash -- the chosen epoch would just be the
    one that memorised best, and its score would look excellent.
    """
    select_on = cfg["eval"].get("select_on", "val")
    trained = set(cfg["data"].get("train_extra") or [])
    if select_on not in ("val", "seal", "none"):
        raise KeyError(f"unknown eval.select_on {select_on!r}; have 'val', 'seal', 'none'")
    if select_on in trained:
        raise ValueError(f"eval.select_on={select_on!r} but {select_on} is in data.train_extra; "
                         f"choose weights on 'seal' or keep the final ones with 'none'")
    if select_on == "seal" and not cfg["data"].get("exclude_seal", True):
        raise ValueError("eval.select_on='seal' needs data.exclude_seal=True -- the seal "
                         "cannot choose the weights of a model that trained on it")
    return select_on


def _fit(cfg, out_dir, verbose: bool):
    """Train a model, choosing weights by the real score on the selection set."""
    from .train import train          # imported late: train.py imports this module

    train_dataset = build_training_dataset(cfg, verbose)
    select_on = selection_set(cfg)
    traj_ids = None
    if select_on == "val":
        split, eval_dataset = "val", build_dataset(cfg, "val")
    elif select_on == "seal":
        # The seal lives inside train, so it is scored as a named subset of that
        # split -- the same path score_seal() takes.
        split = cfg["data"]["train_split"]
        rows = load_splits().seal_rows(load_cache(split))
        eval_dataset = build_dataset(cfg, split, traj_subset=rows)
        traj_ids = [str(eval_dataset.cache.trajectories.traj_id.iloc[t]) for t in rows]
    window_ids = window_ids_of(eval_dataset) if select_on == "seal" else None

    def evaluate(candidate) -> dict:
        table, stats = predict_dataset(candidate, eval_dataset, cfg,
                                       window_ids=window_ids, verbose=False)
        result = score_submission(table, split, cross_check=False, traj_ids=traj_ids)
        if stats["gravity_angle_deg"]:
            # Logged beside the score every epoch; model selection still reads
            # only the score.
            result["gravity_angle_deg"] = stats["gravity_angle_deg"]["macro_median"]
        return result

    fitted = train(cfg, train_dataset, None if select_on == "none" else evaluate,
                   out_dir, verbose)
    return fitted["model"], {"best": fitted["best"], "epochs": fitted["history"]}


def _bar_context(score: float) -> str:
    """Where this score sits among the reference points on val.

    Two of these are bars and they mean different things. 0.8346 is what the
    released baseline scores *legally*, with no platform routing -- beating it is
    the M5 question. 0.4185 is what the same baseline scores when it is told the
    platform, which the rules now forbid; it is a target for internal
    conditioning to aim at, not a score anyone gets for free.
    """
    return (f"reference points on val: floor {VAL_FLOOR:.4f} | "
            f"routed baseline (target, illegal to submit) {VAL_BASELINE_ROUTED:.4f} | "
            f"legal baseline {VAL_BASELINE_FORCED:.4f} | "
            f"all zeros {VAL_ZEROS:.4f}\n"
            f"                    ->  this run {score:.4f}")


def _results_record(cfg, split, result, stats, seconds, out_dir,
                    history=None, seal=None) -> dict:
    """The results.jsonl line. Schema fixed at M0; PipelinePlan.md §10."""
    env = env_record()
    metrics = None
    if result:
        metrics = {
            "macro_ave": result["macro_ave"],
            "macro_ate20": result["macro_ate20"],
            "score": result["score"],
            "official_score": result.get("official_score"),
            "per_platform": result["platforms"],
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "milestone": cfg["run"]["milestone"],
        "run_name": cfg["run"]["name"],
        "notes": cfg["run"]["notes"],
        "config_hash": cfg.hash,
        "reference_name": cfg.reference_name,
        "reference_version": cfg.reference_version,
        "config_diff": cfg.diff,
        "git_sha": env["git_sha"],
        "git_dirty": env["git_dirty"],
        "seed_weights": cfg["run"]["seed_weights"],
        "seed_data": cfg["run"]["seed_data"],
        "seed_augment": cfg["run"]["seed_augment"],
        # Stamped so a ceiling experiment can never be quietly compared against
        # a real run. See data/cond.py.
        "diagnostic_only": cfg["run"]["diagnostic_only"],
        "cond_source": cfg["cond"]["source"],
        "model_kind": cfg["model"]["kind"],
        "n_parameters": stats["n_parameters"],
        "input_repr": cfg["data"]["input_repr"],
        "chunk_len": cfg["data"]["chunk_len"],
        "chunk_stride": cfg["data"]["chunk_stride"],
        "sampler_mode": cfg["sampler"]["mode"],
        "post_ops": list(cfg["post"]["ops"]),
        "split": split,
        "metrics": metrics,
        # Fraction of ATE20 segments fully contained in a chunk, per platform.
        # Null when the segment cache is not loaded -- the ATE20 term is off, so
        # nothing is being excluded from anything.
        "ate20_segment_coverage": stats["segment_coverage"],
        "n_chunks": stats["n_chunks"],
        "chunk_coverage": stats["coverage"],
        "train_loss_proxy": stats["train_loss_proxy"],
        # M15: the gravity head's angle to truth on this split, per platform
        # median and their macro mean. Null for every model without the head.
        "gravity_angle_deg": stats.get("gravity_angle_deg"),
        # The sealed holdout, when this run opened it. Read as a change across
        # model versions, never as a level -- see score_seal().
        "seal": ({"score": seal["score"], "macro_ave": seal["macro_ave"],
                  "macro_ate20": seal["macro_ate20"],
                  "per_platform": seal["platforms"]} if seal else None),
        # Epoch count and best epoch, so a results line says whether a run
        # stopped early and where its chosen weights came from.
        "epochs_run": (len(history["epochs"]) if history else 0),
        "best_epoch": (history["best"]["epoch"] if history else None),
        "best_weights": (history["best"]["which"] if history else None),
        "wall_clock_s": round(seconds, 2),
        "device": cfg["run"]["device"],
        "run_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- CLI

def _parse_set(pairs: list[str]) -> dict:
    """`--set data.chunk_len=64` -> nested override dict, values parsed as JSON."""
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", default="m0_zero")
    ap.add_argument("--split", default=None)
    ap.add_argument("--set", action="append", dest="sets", default=[],
                    metavar="KEY=VALUE", help="dotted config override, JSON value")
    ap.add_argument("--name", default=None)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    over = _parse_set(a.sets)
    if a.name:
        over.setdefault("run", {})["name"] = a.name
    run(a.reference, over, a.split, a.rerun)


if __name__ == "__main__":
    main()
