"""One resolved configuration object per run, hashed. PipelinePlan.md §5.

Three rules, all enforced here rather than by convention:

* **Defaults live in one place.** A run config is a *diff* against a named,
  versioned reference config, and that diff is what the run log prints, so one
  glance shows what an experiment actually changed.
* **The config hash excludes `run.name` and `run.notes` and includes everything
  else.** Two runs with the same science get the same hash, which gives
  duplicate detection for free.
* **The reference is versioned.** When a reference is promoted after a
  bake-off, the old one keeps its name and version, and every results.jsonl
  line records which reference it diffed from. Without this, ablation tables
  drift and the report cannot be reconstructed.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .metric_constants import W_ATE_ABS, W_AVE_ABS
from .utils import stable_hash # a function that turns a bundle of settings into a short fingerprint string.

# Keys excluded from the hash. Everything else is included.
HASH_EXCLUDE = (("run", "name"), ("run", "notes"), ("run", "checkpoint_every")) # This is a list of two "addresses." Each address has two parts. ("run", "name") means "inside the run section, the name box." ("run", "notes") means "inside the run section, the notes box." These two boxes are the ones the fingerprint will ignore, exactly as rule two requires.


# --------------------------------------------------------------------------- merging

def _is_index_patch(v) -> bool:
    """Is `v` a `{"1": {...}}`-shaped override, meant to patch one list element?"""
    return (isinstance(v, Mapping) and len(v) > 0
            and all(isinstance(k, str) and k.isdigit() for k in v))


def deep_merge(base: Mapping, over: Mapping) -> dict:
    """Recursive dict overlay. Lists replace wholesale unless patched by index.

    The one exception exists so that a single element of an ordered list can be
    moved by a dotted key: `loss.terms.1.enabled = true` switches the ATE20 term
    on without restating the whole list, which is what lets an ablation rung be
    expressed as `--key loss.terms.1.enabled --values true false` and read back
    as a two-value column in the report rather than as two walls of JSON. A
    whole-list override still replaces, because a list of different length is a
    different list, not a patch.

    Patching the index one past the end *appends*, so an auxiliary loss term can
    be added the same way: `loss.terms.3={"name": "gravity", "weight": 0.1}`.
    Anything further out is still an error -- a gap in an ordered list is
    always a typo.
    """
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        elif _is_index_patch(v) and isinstance(out.get(k), list):
            merged = copy.deepcopy(out[k])
            for i, patch in sorted(v.items(), key=lambda kv: int(kv[0])):
                idx = int(i)
                if idx == len(merged):
                    merged.append(copy.deepcopy(patch))
                    continue
                if idx > len(merged):
                    raise IndexError(
                        f"override patches element {idx} of {k!r}, which has only "
                        f"{len(merged)} elements")
                merged[idx] = (deep_merge(merged[idx], patch)
                               if isinstance(merged[idx], Mapping)
                               and isinstance(patch, Mapping) else copy.deepcopy(patch))
            out[k] = merged
        else:
            out[k] = copy.deepcopy(v)
    return out


def diff_against(base: Mapping, resolved: Mapping, _prefix: str = "") -> dict[str, Any]:
    """Flat dotted-key diff of `resolved` against `base` — what this run changed.

    A list of dicts of unchanged length is compared element by element, so
    switching one loss term on reads as `loss.terms.1.enabled` in the run log
    rather than as the entire list of terms. Any other list difference is
    reported whole: it is a replacement, not an edit.
    """
    out: dict[str, Any] = {}
    for k, v in resolved.items():
        key = f"{_prefix}{k}"
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            out.update(diff_against(base[k], v, f"{key}."))
        elif (isinstance(v, list) and isinstance(base.get(k), list)
              and len(v) == len(base[k])
              and all(isinstance(a, Mapping) and isinstance(b, Mapping)
                      for a, b in zip(base[k], v))):
            for i, (b_el, v_el) in enumerate(zip(base[k], v)):
                out.update(diff_against(b_el, v_el, f"{key}.{i}."))
        elif k not in base or base[k] != v:
            out[key] = v
    return out


# --------------------------------------------------------------------- reference configs

_M0_ZERO: dict[str, Any] = {
    "run": {
        "name": "m0-zero", #! excluded from hash
        "notes": "M0 plumbing check: constant-zero model through the real pipeline.", #! excluded
        "milestone": "M0",
        "seed_weights": 42,
        "seed_data": 42,
        "seed_augment": 42,
        # Gate for anything that reads a ground-truth label into an inference
        # path. Stamped into every result so a ceiling experiment can never be
        # quietly compared against a real one.
        "diagnostic_only": False,
        # Kept inside the hash on purpose: the device changes the numerics, so
        # two runs that differ only by device are not the same run.
        "device": "cpu",
    },
    "data": {
        "splits": ["val"],
        "train_split": "train",
        "exclude_seal": True,      # nothing may ever be fitted on the sealed holdout
        "chunk_len": 256,          # a MAXIMUM, not a fixed size (PipelinePlan §2.2)
        "chunk_stride": 128,       # chunk_len / 2 at inference, for overlap averaging
        "buckets": [64, 128, 256],
        "snap_to_segments": False,  # needs prep.segments (M1)
        "input_repr": "body",      # 6 channels; body_grav becomes the default at M5
        "extra_scalars": [],
        "normalization": "none",
        "batch_size": 16,          # 4 platforms x 4 chunks once the sampler lands (M11)
    },
    "prep": {
        "orientation": {"mode": "none", "gains": {}, "adaptive": False},
        "gravity": {"source": "filter"}, 
        "segments": {"enabled": False},
    },
    "cond": {
        # M0 has no classifier yet, so the documented sentinel is used: a uniform
        # 0.25 vector. This reads no label, so it is NOT a diagnostic path.
        "source": "uniform",
        "classifier_ckpt": None,
        "corruption_rate": 0.0,
    },
    "model": {
        "kind": "constant_zero",
        "encoder": {"mode": "none"},
        "context": {"mode": "identity", "direction": "bidirectional"},
        "conditioner": {"mode": "none", "insert_at": []},
        "heads": {"velocity": {"mode": "constant"}},
    },
    "loss": {
        # Ordered terms with ABSOLUTE weights, so the loss is the score up to an
        # additive constant (PipelinePlan §4.4).
        "terms": [
            {"name": "ave", "weight": W_AVE_ABS},
            {"name": "ate20", "weight": W_ATE_ABS, "enabled": False},  # slot; built at M10
        ],
    },
    "optim": {
        "lr": 1e-3, "schedule": "none", "epochs": 0,
        "amp": False, "grad_clip": 0.0,
        "weight_decay": 0.0, "warmup_fraction": 0.05,
        "ema_decay": 0.0, "patience": 0,
    },
    "sampler": {
        "mode": "sequential",      # platform_balanced becomes the default at M11
        "platform_weights": None,
        "within_platform_weighting": "none",
    },
    "augment": {"ops": []},
    "post": {"ops": ["stitch"], "stitch": {"weighting": "uniform"}},
    "eval": {"splits": ["val"], "cadence": "end", "write_submission": True,
             "seal": False},
}


# --------------------------------------------------------------------- M5: first model
#
# The reference config from models.md §9, and the thing every later ablation
# moves exactly one line of. Nothing in it is arbitrary and nothing in it is
# settled; each line has a milestone attached:
#
#   encoder mode        -> M6   backbone bake-off
#   input_repr          -> M7   gravity alignment and the aliasing question
#   context mode        -> M8   crossed with chunk_len
#   conditioner mode    -> M9   the headline lever
#   loss.ate20          -> M10  the metric-matched trajectory term
#   sampler mode        -> M11  batch composition
#
# Two departures from models.md §9, both deliberate and both recorded here so a
# future session sees them rather than rediscovering them:
#
# 1. `batch_size` is 256, not 16. §9 pairs `chunk_len: 1` with 4 chunks per
#    platform, which would make an optimiser step out of sixteen single windows
#    -- 4,000+ tiny steps per epoch, and a benchmark measured at a shape 32x
#    larger. 64 chunks per platform keeps the windows-per-step near the 512 the
#    architecture survey actually timed. When chunk_len rises at M8 this must
#    come back down in proportion.
# 2. `input_repr` is `body_grav` as §9 says, and the gravity channels it needs
#    are now built -- but note the measured caveat in prep/orientation.py: the
#    filter lands within about 1 degree of truth on car, dog and human, and is
#    much less reliable on drone (median 7-9 degrees, a third of trajectories
#    beyond 15). Whether those channels help or hurt is exactly what M7 asks.

_M5_TWO_STAGE: dict[str, Any] = deep_merge(_M0_ZERO, {
    "run": {
        "name": "m5-two-stage",
        "notes": "M5: first own model -- window encoder, internal embodiment "
                 "branch, concat conditioning. Bar to beat: 0.8346 on val.",
        "milestone": "M5",
        # In the hash on purpose: the device changes the numerics, so two runs
        # differing only by device are not the same run. Measured on this
        # MacBook, `mps` runs a step in 42 ms against `cpu`'s 223 ms -- an epoch
        # is 24 s instead of 105 s. Override with
        #     --set run.device='"cpu"'
        # on a machine without a Metal GPU.
        "device": "mps",
    },
    "data": {
        "chunk_len": 1,            # -> will increase in M8, when context has something to read
        "chunk_stride": 1,
        "buckets": [1],
        "input_repr": "body_grav",  # 9 channels: raw IMU + per-frame gravity direction
        "normalization": "dataset_fixed",   # never per-window; see prep/norm.py
        "batch_size": 256,         # 4 platforms x 64 chunks
    },
    "cond": {
        # Nothing enters the batch. The model builds its own embodiment vector
        # from the signal -- the 2026-08-30 rules clarification requires it.
        "source": "learned",
    },
    "model": {
        "kind": "two_stage",
        "encoder": {"mode": "cnn_small", "d_model": 256},
        "context": {"mode": "identity"},
        "embodiment": {"pool": "mean", "width": 32},
        "conditioner": {"mode": "concat"},
        "heads": {"velocity": True, "platform": True},
        "dropout": 0.1,
    },
    "loss": {
        "terms": [
            {"name": "ave", "weight": W_AVE_ABS},
            {"name": "ate20", "weight": W_ATE_ABS, "enabled": False},   # M10
            # Auxiliary only. Breaks the "loss == score" property, which is why
            # the bundle reports score_proxy separately from total.
            {"name": "platform", "weight": 0.1},
        ],
    },
    "optim": {
        # 60 epochs at 371 steps each is about 26 minutes on this laptop's GPU.
        # models.md §8 says 100; 60 is what fits three seeds into one sitting,
        # and early stopping ends a run sooner whenever it can.
        "lr": 3e-4, "schedule": "cosine", "epochs": 60,
        "weight_decay": 0.01, "grad_clip": 1.0,
        "warmup_fraction": 0.05, "ema_decay": 0.999, "patience": 12,
    },
    "sampler": {"mode": "platform_balanced"},
    "eval": {"splits": ["val"], "seal": True},
})

# ------------------------------------------------- M8: context, and the first
#                                                    thing that moved drone
#
# Promoted from the M8 bake-off on 2026-09-04. `m5_two_stage` keeps its name and
# version and every earlier results.jsonl line still points at it -- that is the
# rule this module's docstring states, and it is what lets the report's ablation
# tables be rebuilt from the log months later.
#
# Five lines move, and they move together rather than independently:
#
#   context.mode: identity -> bigru
#       The measured lever. Worth -0.1032 against a matched `identity` control at
#       the same chunk length, 28x the noise floor. It beat `transformer` by
#       0.0725 with 2.8x fewer parameters.
#
#   chunk_len: 1 -> 64, and batch_size: 256 -> 4
#       These two are one decision. 64 is the drone-first choice: every drone
#       trajectory is at most 60 windows, so a whole flight lands in one chunk --
#       and no longer chunk can give drone any more context than it already has.
#       `batch_size` falls in step to keep 256 window slots per optimiser step, so
#       the comparison against M5 is not also a batch-size study.
#
#   epochs: 60 -> 150, patience: 12 -> 20
#       Not cosmetic. M5's 60 epochs was tuned at chunk_len 1 and does NOT
#       transfer: 4 chunks of 64 consecutive windows is a much noisier gradient
#       than 256 unrelated windows, so convergence takes far longer. At 60 epochs
#       the run was still improving by 0.0035 in its last fifth -- at the noise
#       floor. At 150 it is down to 0.0018, below it. The extra epochs are worth
#       0.0280, which is 7.6x the noise floor and would have been missed entirely.
#
# The cost is real and should be known before this is used: one run is about
# three hours on this laptop's GPU, against 24 minutes for `m5_two_stage`.

_M8_BIGRU: dict[str, Any] = deep_merge(_M5_TWO_STAGE, {
    "run": {
        "name": "m8-bigru",
        "notes": "M8: bidirectional context over 64-window chunks. The first "
                 "configuration to move drone -- AVE 0.715 -> 0.461.",
        "milestone": "M8",
    },
    "data": {
        "chunk_len": 64,        # a whole drone flight, and no more padding than needed
        "chunk_stride": 64,     # no overlap: halving it doubles cost and gives drone nothing
        "buckets": [64],
        "batch_size": 4,        # 4 x 64 = the same 256 window slots per step as M5
    },
    "model": {"context": {"mode": "bigru", "direction": "bidirectional"}},
    "optim": {"epochs": 150, "patience": 20},
    # The seal stays shut. `m5_two_stage` inherits `seal: True` from the M4 run
    # that first opened it, and carrying that forward would mean every routine
    # run from here silently spends an opening -- the log already stands at five
    # against PipelinePlan.md §6.6's budget of three. M12 turns it back on
    # deliberately, and that is the only place that should.
    "eval": {"seal": False},
})


# --------------------------------------------------------------------------- m9_film
#
# Promoted from the M9 conditioner bake-off on 2026-09-06, at three seeds:
# 0.2542 +/- 0.0016 against `m8_bigru`'s 0.2646 +/- 0.0016, a gap of 0.0103 or
# 3.2x the 0.0032 seed floor. `m8_bigru` keeps its name and version, so every
# earlier results.jsonl line still resolves and the ablation tables rebuild.
#
# One line moves:
#
#   conditioner.mode: concat -> film
#       The embodiment vector stops being 32 extra numbers glued onto a 256-dim
#       feature and starts setting a per-channel scale and offset on it. Drone
#       AVE 0.468 -> 0.435, and both halves of the metric improved together --
#       AVE 0.1986, ATE20 0.7186 -- which is the opposite of M10, where they
#       traded against each other.
#
# Two facts that matter when reading any comparison against this reference:
#
#   It is the SMALLER model. 1,378,279 parameters against concat's 1,417,383,
#   because FiLM's 32->64->512 side network is cheaper than concat's 288x256 mix
#   layer. A win here cannot be explained as extra capacity.
#
#   FiLM's output layer is zero-initialised, so training starts as a bit-exact
#   identity to no conditioning at all (measured: max |diff| 0.000e+00). The
#   ablation therefore compares conditioning that was *learned* against a
#   different mechanism, not against a different initialisation.
#
# What it did NOT do, which is the more useful half for planning. Finding 14
# funded this arm on drone *direction* being the largest lever at 0.0535. After
# film that lever is still worth 0.0511 -- it moved 4.4%. The gain is broad
# rather than targeted: drone's median angle fell 14.7 -> 13.6 deg and its speed
# ratio rose 0.925 -> 0.947, while every platform's speed ratio rose by the same
# +0.021. Drone direction remains the largest single lever on the board.

_M9_FILM: dict[str, Any] = deep_merge(_M8_BIGRU, {
    "run": {
        "name": "m9-film",
        "notes": "M9: FiLM conditioning on the embodiment vector. 0.2542 at "
                 "three seeds, -0.0103 against concat, with fewer parameters.",
        "milestone": "M9",
    },
    "model": {"conditioner": {"mode": "film"}},
})


# --------------------------------------------------------------------------- m7_body
#
# NOT a promotion. `m9_film` with its three gravity channels dropped -- the
# six-channel negative control that CLAUDE.md finding 15 rests on, 0.2721 +/-
# 0.0040 at three seeds on a Kaggle T4, against `m9_film`'s 0.2542.
#
# It is named because M15 needs a baseline to sit on: the gravity head is the
# candidate *substitute* for those channels, so it trains on six channels and is
# measured against this. The three `m7-body` runs were launched from the Kaggle
# notebook as `m9_film` plus overrides, before this name existed, and this entry
# resolves to exactly their science -- with `run.device = "cuda"` and the seed
# keys set it reproduces #bb699b34, #61fba7cc and #301ecf60 (tests/test_m15.py),
# so those three lines and every future run against this name are one arm.
#
# The milestone is part of that: it sits inside the hash, and those runs said M7.

_M7_BODY: dict[str, Any] = deep_merge(_M9_FILM, {
    "run": {
        "name": "m7-body",
        "notes": "M7: m9_film on six raw channels, no gravity channels. The "
                 "negative control: 0.2721 at three seeds, +0.0179 against m9_film.",
        "milestone": "M7",
    },
    "data": {"input_repr": "body"},
})


REFERENCES: dict[str, dict[str, Any]] = {
    "m0_zero": {"version": 1, "config": _M0_ZERO},
    "m5_two_stage": {"version": 1, "config": _M5_TWO_STAGE},
    "m8_bigru": {"version": 1, "config": _M8_BIGRU},
    "m9_film": {"version": 1, "config": _M9_FILM},
    "m7_body": {"version": 1, "config": _M7_BODY},
}


# ------------------------------------------------------------ M15: the gravity head
#
# Deliberately an override, not a reference key. Every key is inside the config
# hash, so declaring `heads.gravity: false` in a reference would rehash every
# run in results.jsonl; instead the model treats an absent key as off, and this
# function is the one canonical way to switch it on. The canonical form matters:
# `{"name": "gravity", "weight": 0.1}` and the same dict with `"enabled": True`
# are the same science but different hashes, so everything -- the CLI, the
# Kaggle notebook, the tests -- should build it here.
#
# The command-line equivalent, against any reference with three loss terms:
#     --set model.heads.gravity=true
#     --set 'loss.terms.3={"name": "gravity", "weight": 0.1}'

GRAVITY_TERM_WEIGHT = 0.1     # matches the platform term; GravityDirectionPlan.md


def gravity_head_overrides(reference: str,
                           weight: float = GRAVITY_TERM_WEIGHT) -> dict[str, Any]:
    """Overrides that switch on the gravity head and append its loss term."""
    n_terms = len(REFERENCES[reference]["config"]["loss"]["terms"])
    return {
        "model": {"heads": {"gravity": True}},
        "loss": {"terms": {str(n_terms): {"name": "gravity", "weight": weight}}},
    }


# ---------------------------------------------------------- M17: augmentation
#
# An override, not a reference key, for the same reason the gravity head is one:
# every key sits inside the config hash, so declaring `augment.remount` in a
# reference would rehash every run already in results.jsonl and the report's
# ablation tables would stop resolving. `data/augment.build_augmenter` treats an
# absent key -- and an empty `ops` list, which every existing reference has --
# as off.
#
# The command-line equivalent:
#     --set 'augment={"ops": ["remount"], "remount": {"recipe": "remount_v1"}}'

AUGMENT_RECIPE = "remount_v1"     # .claude/augmentation.md §4


def augment_overrides(recipe: str = AUGMENT_RECIPE, **tweaks: Any) -> dict[str, Any]:
    """Overrides that switch augmentation on at a named recipe.

    `tweaks` override individual recipe fields (`yaw_deg`, `tilt_deg`,
    `mirror_p`) and are written into the config, so the run log and the config
    hash both record the exact magnitudes rather than just the recipe's name.
    """
    from .data.augment import RECIPES

    if recipe not in RECIPES:
        raise KeyError(f"unknown augment recipe {recipe!r}; have {sorted(RECIPES)}")
    return {"augment": {"ops": ["remount"],
                        "remount": {"recipe": recipe, **RECIPES[recipe], **tweaks}}}


# ------------------------------------------------------- M22: time scaling
#
# An override for the same reason: every key sits in the hash. The recipe's
# numbers are written out in full, like the re-mounting magnitudes, so the hash
# and the run log record exactly what speeds were drawn and how often.
#
# The command-line equivalent:
#     --set 'augment={"ops": ["timescale"], "timescale": {"recipe": "timescale_drone"}}'

TIMESCALE_RECIPE = "timescale_drone"     # .claude/augmentation.md §17


def timescale_overrides(recipe: str = TIMESCALE_RECIPE, **tweaks: Any) -> dict[str, Any]:
    """Overrides that switch time scaling on at a named recipe.

    `tweaks` override recipe fields -- `s_range`, `p`, `fc_hz`,
    `split_box_seconds` -- per platform dictionaries replacing the recipe's
    whole dictionary, so what is written is exactly what runs. `phase_p` (M23,
    per platform) adds the random window-grid shift; absent, nothing changes.
    """
    from .data.timescale import RECIPES

    if recipe not in RECIPES:
        raise KeyError(f"unknown timescale recipe {recipe!r}; have {sorted(RECIPES)}")
    return {"augment": {"ops": ["timescale"],
                        "timescale": {"recipe": recipe, **RECIPES[recipe], **tweaks}}}


# ------------------------------------------------ M19: the gravity estimator
#
# An override, not a reference key, for the reason above: every key sits in the
# hash. `prep.orientation.estimator_spec` treats an absent `estimator` as the
# causal filter every run through M18 used, so no earlier line in results.jsonl
# rehashes. The estimator's parameters are written out in full, like the
# augmentation magnitudes, so the hash records exactly what made the channels.
#
# The command-line equivalent:
#     --set 'prep.gravity={"source": "filter", "estimator": "box", "box_seconds": 5.0}'

def gravity_estimator_overrides(estimator: str, **params: Any) -> dict[str, Any]:
    """Overrides that switch the three gravity channels to another estimator."""
    from .prep.orientation import estimator_spec

    return {"prep": {"gravity": estimator_spec({"estimator": estimator, **params})}}


# ------------------------------------------- M16 / M20: the ground-truth oracle
#
# `noise_deg = 0` is M16's oracle exactly -- the same two keys its notebook set,
# so the same hash. `noise_deg > 0` is M20's dose-response: truth plus a slowly
# drifting tilt error with that median (prep.orientation.degrade_gravity).
# Either way the run reads ground truth into the model, so it is stamped
# diagnostic-only and can never predict test.

TRUTH_NOISE_TAU_S = 3.0     # seconds; a filter's error drifts over seconds, not frames


def truth_gravity_overrides(noise_deg: float = 0.0,
                            tau_s: float = TRUTH_NOISE_TAU_S) -> dict[str, Any]:
    """Overrides for the ground-truth gravity oracle, optionally degraded."""
    g: dict[str, Any] = {"source": "truth"}
    if noise_deg > 0:
        g.update({"truth_noise_deg": float(noise_deg), "truth_noise_tau_s": float(tau_s)})
    return {"run": {"diagnostic_only": True}, "prep": {"gravity": g}}


# ------------------------------------------------ M21: AirIO's attitude branch
#
# An override, not a reference key, for the same reason as the rest of this
# section. `models.two_stage` treats an absent `model.attitude` as off.
#
# The command-line equivalent:
#     --set 'model.attitude={"mode": "late", "width": 64}'

def attitude_branch_overrides(mode: str = "late", width: int = 64) -> dict[str, Any]:
    """Overrides that switch on AirIO's separate attitude branch (models.two_stage)."""
    from .models.two_stage import ATTITUDE_MODES

    if mode not in ATTITUDE_MODES:
        raise KeyError(f"unknown attitude mode {mode!r}; have {list(ATTITUDE_MODES)}")
    return {"model": {"attitude": {"mode": mode, "width": int(width)}}}


# ------------------------------------------------- M29: continuing a fitted run
#
# An override, not a reference key, like everything else in this section: adding
# `optim.init_from` to a reference would rehash every run already in
# results.jsonl. `train.train` treats an absent key as a cold start.
#
# Two keys, doing two different jobs:
#
#   optim.init_from    WHICH weights this run starts from. It is inside the
#                      config hash, because a run that starts somewhere is not
#                      the same experiment as one that starts from noise. It is
#                      a path, so the same science hashed on Kaggle and on the
#                      laptop gives two hashes -- true of `run.device` already,
#                      and the price of recording the provenance honestly.
#   run.checkpoint_every  HOW OFTEN a long run saves something recoverable.
#                      Bookkeeping, so it is excluded from the hash: switching
#                      it on leaves every existing hash exactly where it was.
#
# The command-line equivalent:
#     --set optim.init_from=runs/<run>/model_ema.pt --set optim.epochs=60
#     --set optim.lr=0.0005 --set run.checkpoint_every=10

def warm_start_overrides(init_from: str, epochs: int, lr: float | None = None,
                         weights: str = "ema", restore: bool = False,
                         checkpoint_every: int = 10) -> dict[str, Any]:
    """Overrides that continue training from weights an earlier run finished with.

    `epochs` is this run's OWN budget, and the cosine schedule is laid out over
    it from scratch: the learning rate warms up to `lr`, then decays to zero
    across those epochs. That restart is the point -- a run that has flattened
    out under a decayed learning rate is being asked whether a fresh climb finds
    anywhere better, which is the warm-restart idea (SGDR) rather than a resume.

    `restore=True` is the other thing, and it is for rescue rather than for
    science: it carries the optimiser's moments and the moving average across so
    that an interrupted run continues as nearly as it can. It needs a
    `checkpoint.pt`, since a `model_*.pt` holds weights alone.
    """
    if weights not in ("ema", "live"):
        raise KeyError(f"unknown weights {weights!r}; have 'ema', 'live'")
    # `init_weights` is stated only when it is not the default, for the same
    # reason the notebook states `prep.gravity.source` only when it moves: an
    # explicit "ema" and an absent one are the same science and must not be two
    # hashes. Without this the README's command line and this builder would
    # disagree on the hash of the same run.
    optim: dict[str, Any] = {"init_from": str(init_from), "epochs": int(epochs)}
    if weights != "ema":
        optim["init_weights"] = weights
    if lr is not None:
        optim["lr"] = float(lr)
    if restore:
        optim["init_restore"] = True
    return {"optim": optim, "run": {"checkpoint_every": int(checkpoint_every)}}


def trainval_overrides(select_on: str, epochs: int = 150) -> dict[str, Any]:
    """Overrides that add val to the training data (D9), in one of two shapes.

    ``"seal"``: train + val, the seal held out and scored every epoch. No early
    stop, so the whole schedule's curve is on record and "best epoch" can be read
    against "final EMA" -- the rule the other shape relies on.

    ``"none"``: train + seal + val, nothing held out, every epoch run, the final
    EMA weights kept. Across seven m9_film-family runs the final EMA sat 0.0003 to
    0.0026 from the best epoch, inside the 0.0037 noise floor.
    """
    if select_on == "seal":
        return {"data": {"train_extra": ["val"]},
                "eval": {"select_on": "seal", "seal": True},
                "optim": {"epochs": int(epochs), "patience": int(epochs)}}
    if select_on == "none":
        return {"data": {"train_extra": ["val"], "exclude_seal": False},
                "eval": {"select_on": "none"},
                "optim": {"epochs": int(epochs)}}
    raise KeyError(f"unknown select_on {select_on!r}; have 'seal', 'none'")


# --------------------------------------------------------------------------- hashing

def _strip_for_hash(cfg: Mapping) -> dict:
    out = copy.deepcopy(dict(cfg))
    for section, key in HASH_EXCLUDE:
        if section in out and isinstance(out[section], dict):
            out[section].pop(key, None)
    return out


def config_hash(cfg: Mapping, n: int = 8) -> str:
    return stable_hash(_strip_for_hash(cfg), n)


# --------------------------------------------------------------------------- entrypoint

class Config(dict):
    """A resolved config, plus the provenance a results.jsonl line needs."""

    def __init__(self, resolved: Mapping, reference: str, version: int,
                 diff: Mapping):
        super().__init__(resolved)
        self.reference_name = reference
        self.reference_version = version
        self.diff = dict(diff)
        self.hash = config_hash(resolved)

    def section(self, name: str) -> dict:
        return self[name]

    def provenance(self) -> dict:
        return {
            "config_hash": self.hash,
            "reference_name": self.reference_name,
            "reference_version": self.reference_version,
            "config_diff": self.diff,
        }


def resolve(reference: str = "m0_zero", overrides: Mapping | None = None) -> Config:
    """Resolve `reference` + `overrides` into one hashed Config."""
    if reference not in REFERENCES:
        raise KeyError(f"unknown reference config {reference!r}; "
                       f"have {sorted(REFERENCES)}")
    ref = REFERENCES[reference]
    base = ref["config"]
    resolved = deep_merge(base, overrides or {})
    return Config(resolved, reference, ref["version"], diff_against(base, resolved))
