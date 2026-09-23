"""The training loop. models.md §8.

    mySolution/.venv/bin/python -m src.pipeline --reference m5_two_stage --train

This is the first place in the project where anything is actually *fitted*.
Everything before it -- the caches, the sample contract, the losses, the scorer
-- exists so that this loop can be short and boring, which is what a training
loop should be.

**What one epoch means here, and why it is not "one pass over the data".**
Batches are built with an equal number of chunks from each of the four
platforms, because the competition score weights the four platforms equally. The
platforms have wildly different amounts of data, so keeping the batches balanced
means the small platforms get revisited within a single epoch while the large
ones are not exhausted. An epoch is therefore a fixed number of balanced steps.
That is the intended behaviour; see `data/sampler.py` for the numbers that force
it.

**What is watched, and what is optimised, are different things.** The optimiser
minimises the weighted sum of the velocity error and the auxiliary platform
error. The *decision* about which weights to keep is made on the real
competition score, computed on the val split by the same scorer the leaderboard
uses. Those are not the same number and should not be confused: the auxiliary
term deliberately trades a little velocity accuracy for a better-organised
embodiment vector, so the training loss can rise while the score improves.

**Weight averaging.** Alongside the weights being trained, a slowly-moving
average of them is kept. It lags behind, which means it is not thrown around by
whichever batch happened to come last, and it usually scores a little better for
free. Both sets are evaluated each epoch and the better one is kept.

**Weights can be picked up again, and they can be picked up mid-flight.**
Two separate things make that work, and they are worth keeping apart. A run may
be *started from* weights another run finished with (`optim.init_from`), which
rebuilds the optimiser from scratch and restarts the learning-rate schedule --
the warm restart you want when a long run has flattened out and the question is
whether a fresh climb finds somewhere better. Separately, a long run may write a
`checkpoint.pt` every few epochs (`run.checkpoint_every`), so a session that is
killed at epoch 200 of 250 leaves its weights behind instead of nothing. Until
that key existed, weights were written once, after the last epoch, and an
interrupted run lost everything it had learned.

**Progress is emitted twice, for two different readers.** A one-line, in-place
step counter goes to whoever is watching the process, and `history.json` is
rewritten in the run directory after every epoch for whoever is not -- a
notebook plotting the curve of a run started in another terminal, or the
post-mortem of a run that was interrupted. Neither costs anything measurable.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch

from .data.sampler import build_batches, platform_share
from .evaluation.score import score_submission
from .losses import build_loss, term_enabled
from .models import build_model, forward_batch
from .utils import write_json


# --------------------------------------------------------------------------- schedule

def learning_rate_at(step: int, total_steps: int, base_lr: float,
                     warmup_fraction: float = 0.05) -> float:
    """Linear warmup, then a cosine decay down to zero.

    The warmup matters more than usual here. At step zero the velocity head is
    effectively random, so the first gradients are large and badly aimed; taking
    full-size steps on them wastes the first part of training undoing the damage.
    Ramping in over the first few percent of steps avoids that for free.
    """
    total_steps = max(total_steps, 1)
    warmup_steps = max(int(total_steps * warmup_fraction), 1)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class WeightAverage:
    """An exponentially-weighted moving average of the model's weights.

    Every step it moves a tiny fraction of the way from its own values toward
    the live ones, so it behaves like a smoothed version of the training
    trajectory. Cheap, and usually worth a small fraction of a score point.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_p, live_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(live_p.detach(), alpha=1.0 - self.decay)
        for shadow_b, live_b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(live_b)


# ------------------------------------------------------------- warm starting

#: What `optim.init_weights` may say when the file holds more than one set.
INIT_WEIGHTS = ("ema", "live")


def load_checkpoint(path) -> dict:
    """Read a saved checkpoint and hand it back in one shape, whatever it was.

    Three kinds of file legitimately end up here:

    * ``model.pt`` / ``model_ema.pt`` / ``model_live.pt`` -- what a finished run
      writes. One set of weights under ``state_dict``, and no optimiser.
    * ``checkpoint.pt`` -- what a run in progress writes every `checkpoint_every`
      epochs. It carries the live weights, the moving average, and the
      optimiser's own state, so a killed session can be picked up rather than
      repeated.
    * a bare ``state_dict`` saved by hand.

    Normalising here means no caller has to know which of the three it was given.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"optim.init_from points at {path}, which is not a file. On Kaggle an "
            f"earlier run's weights arrive under /kaggle/input/<name>/ -- add that "
            f"notebook's output in the Input panel and point at the file inside it.")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        blob = {"state_dict": blob}
    return {**blob, "path": str(path)}


def select_init_weights(blob: dict, prefer: str = "ema") -> tuple[dict, str]:
    """Pick one set of weights out of a checkpoint, and say which one it was.

    A finished run's `model_*.pt` holds exactly one set and `prefer` is moot. A
    mid-run `checkpoint.pt` holds both, and the moving average is the better
    starting point by default for the same reason it usually scores better: it
    is not wherever the last batch happened to throw the weights.
    """
    if prefer not in INIT_WEIGHTS:
        raise KeyError(f"unknown optim.init_weights {prefer!r}; have {list(INIT_WEIGHTS)}")
    if blob.get("ema_state") is None:
        # A `model_*.pt`, which holds one set and does not say which kind it is
        # -- `model_ema.pt` keeps the average under the same `state_dict` key
        # that `model_live.pt` keeps the live weights under. Calling that "live"
        # would be a plain lie half the time, so it gets its own name.
        return blob["state_dict"], "sole"
    return ((blob["ema_state"], "ema") if prefer == "ema"
            else (blob["state_dict"], "live"))


def describe_checkpoint(blob: dict, which: str) -> str:
    """One line naming what a warm start is actually starting from."""
    best = blob.get("best") or {}
    epoch = blob.get("epoch", best.get("epoch"))
    score = best.get("score")
    named = {"ema": "the moving average", "live": "the live weights",
             "sole": "the weights it holds"}[which]
    return (f"{Path(blob['path']).name} [{named}"
            + (f", epoch {epoch}" if epoch is not None else "")
            + (f", {score:.4f}" if isinstance(score, float) else "")
            + "]")


def _load_into(model: torch.nn.Module, state: dict, source: str) -> None:
    """Strict load, with an error that says which knob is wrong.

    Strict is deliberate. A key or shape mismatch means the checkpoint was made
    by a different architecture, and quietly keeping the layers that happen to
    line up would produce a model that trains, scores, and means nothing.
    """
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"the weights in {source} do not fit this model. The warm start has to "
            f"use the SAME architecture it is continuing: check model.encoder.mode, "
            f"model.context, model.conditioner and data.input_repr against the "
            f"config stored in the checkpoint.\n\n{exc}") from exc


def write_checkpoint(path: Path, cfg, model, averaged, optimizer,
                     epoch: int, step: int, best: dict) -> None:
    """Save everything a run needs to be continued, in one file, atomically.

    Written to a neighbouring temporary name and then renamed, because the whole
    point of this file is to survive a process being killed -- and a half-written
    checkpoint is worse than none, since it looks like a usable one.
    """
    payload = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "ema_state": ({k: v.detach().cpu().clone()
                       for k, v in averaged.shadow.state_dict().items()}
                      if averaged is not None else None),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch), "step": int(step),
        "best": dict(best), "config": dict(cfg),
        "n_parameters": model.n_parameters(),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)



# --------------------------------------------------------------------------- the loop

def train(cfg, train_dataset, evaluate, out_dir: Path,
          verbose: bool = True) -> dict:
    """Fit a model and return its history plus the best weights found.

    `evaluate(model) -> dict` is supplied by the caller rather than built here,
    so that this function never has to know how a split is scored. It must
    return at least a `score` key; whatever else it returns is logged.

    `evaluate=None` means nothing is held out: every epoch runs, nothing stops
    early, and the weights kept are the final ones -- the EMA when there is one.
    """
    label = cfg["eval"].get("select_on", "val")
    optim_cfg = cfg["optim"]
    device = torch.device(cfg["run"]["device"])
    model = build_model(cfg, in_channels=train_dataset.n_channels).to(device)
    loss_fn = build_loss(cfg)

    # A warm start, when one is asked for. This happens BEFORE the moving
    # average is constructed, so the average begins where the loaded weights
    # are rather than crawling toward them from a random initialisation -- which
    # would otherwise make the first tens of epochs of an EMA-selected run
    # worthless.
    init_blob, init_which = None, None
    if optim_cfg.get("init_from"):
        init_blob = load_checkpoint(optim_cfg["init_from"])
        state, init_which = select_init_weights(
            init_blob, optim_cfg.get("init_weights", "ema"))
        _load_into(model, state, describe_checkpoint(init_blob, init_which))
        if verbose:
            print(f"warm start from {describe_checkpoint(init_blob, init_which)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]))
    averaged = (WeightAverage(model, float(optim_cfg["ema_decay"]))
                if optim_cfg.get("ema_decay") else None)

    # `init_restore` is the difference between continuing a run and starting a
    # new one from its weights. Off (the default) the optimiser's moments and
    # the schedule both start again, which is the warm RESTART: the learning
    # rate climbs back to `optim.lr` and decays afresh over this run's own
    # epochs. On, the moments and the moving average carry over, so the run
    # resumes as nearly as it can what was interrupted. The schedule restarts
    # either way -- `epochs` here is this run's budget, not the original one.
    if init_blob is not None and optim_cfg.get("init_restore"):
        restored = []
        if init_blob.get("optimizer") is not None:
            optimizer.load_state_dict(init_blob["optimizer"])
            restored.append("optimiser moments")
        if averaged is not None and init_blob.get("ema_state") is not None:
            averaged.shadow.load_state_dict(init_blob["ema_state"])
            restored.append("the moving average")
        if verbose:
            print("  restored " + (", ".join(restored) if restored else
                                   "nothing: that file carries weights only"))

    epochs = int(optim_cfg["epochs"])
    grad_clip = float(optim_cfg.get("grad_clip", 0.0))
    patience = int(optim_cfg.get("patience", epochs))
    # Bookkeeping, not science: it is excluded from the config hash, so switching
    # it on never re-hashes a run or hides it from duplicate detection.
    checkpoint_every = int(cfg["run"].get("checkpoint_every", 0) or 0)

    # Batches are planned once to learn how many steps an epoch takes, which the
    # cosine schedule needs up front. They are re-planned with a new seed every
    # epoch so the model does not see the same groupings twice.
    plan = _plan_batches(cfg, train_dataset, epoch=0)
    steps_per_epoch = len(plan)
    total_steps = steps_per_epoch * epochs
    if verbose:
        share = platform_share(train_dataset, plan)
        print(f"training: {len(train_dataset):,} chunks, {steps_per_epoch} steps/epoch, "
              f"{epochs} epochs, {model.n_parameters():,} parameters")
        print(f"platform share per batch: "
              + "  ".join(f"{p}={v:.3f}" for p, v in sorted(share.items())))

    # What the epoch summary averages. The gravity pair is added only when its
    # term is on, so every run without the head logs exactly what it always did.
    tracked = ["total", "ave", "platform", "platform_accuracy"]
    gravity_on = term_enabled(cfg, "gravity")
    if gravity_on:
        tracked += ["gravity", "gravity_angle_deg"]

    history: list[dict] = []
    best = {"score": float("inf"), "epoch": -1, "which": "live"}
    best_state = copy.deepcopy(model.state_dict())
    # Both sets of weights are always kept, each on its own: model_live.pt and
    # model_ema.pt -- the best of each by the selection score, or the final of each
    # when nothing is held out. model.pt stays what it always was (the overall
    # best, or the final EMA) so every existing reader keeps working.
    best_ema = {"score": float("inf"), "epoch": -1, "which": "ema"}
    best_ema_state = None
    best_live = {"score": float("inf"), "epoch": -1, "which": "live"}
    best_live_state = None
    step = 0

    for epoch in range(epochs):
        model.train()
        started = time.perf_counter()
        # A fresh set of augmentation draws for this pass. A no-op for a dataset
        # that is not augmenting, which is every dataset built before today.
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        batches = _plan_batches(cfg, train_dataset, epoch)
        running = {key: 0.0 for key in tracked}
        running["n"] = 0
        last_drawn = 0.0
        #! inner loop iterates over batches
        for i, indices in enumerate(batches, 1):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(
                    step, total_steps, float(optim_cfg["lr"]),
                    float(optim_cfg.get("warmup_fraction", 0.05)))

            batch = train_dataset.batch(indices)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            outputs = forward_batch(model, batch)
            loss, parts = loss_fn(outputs, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if averaged is not None:
                averaged.update(model)

            for key in tracked:
                if key in parts and parts[key] == parts[key]:      # skip NaN
                    running[key] += parts[key]
            running["n"] += 1
            step += 1

            # An epoch is minutes long and used to print nothing until it ended,
            # which made a run indistinguishable from a hang. Redraw one line in
            # place, throttled to four times a second so that a notebook's
            # output area is not the thing being benchmarked.
            now = time.perf_counter()
            if verbose and now - last_drawn > 0.25:
                last_drawn = now
                print(f"\r  epoch {epoch:3d}  step {i:4d}/{steps_per_epoch}"
                      f"  loss {running['total'] / running['n']:.4f}"
                      f"  {i / (now - started):5.1f} steps/s",
                      end="", flush=True)

        train_metrics = {k: (v / max(running["n"], 1))
                         for k, v in running.items() if k != "n"}

        # Model selection runs on the real competition score, never on the loss.
        candidates = {"live": model}
        if averaged is not None:
            candidates["ema"] = averaged.shadow
        if evaluate is None:
            scored, which = {}, ("ema" if averaged is not None else "live")
            best = {"score": None, "epoch": epoch, "which": which}
        else:
            if verbose:
                print(f"\r  epoch {epoch:3d}  scoring {label} ...", end="", flush=True)
            scored = {name: evaluate(m) for name, m in candidates.items()}
            which = min(scored, key=lambda n: scored[n]["score"])

        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "train": {k: round(v, 5) for k, v in train_metrics.items()},
            "val": {name: {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in r.items() if k != "platforms"}
                    for name, r in scored.items()},
            "best_of": which,
        }
        history.append(record)

        if evaluate is None or scored[which]["score"] < best["score"]:
            if evaluate is not None:
                best = {"score": scored[which]["score"], "epoch": epoch, "which": which}
            best_state = copy.deepcopy(candidates[which].state_dict())
        if evaluate is None:
            # Nothing held out: the "best" of each is simply its latest state. Kept
            # every epoch, so an interrupted run still leaves both behind in memory.
            best_live = {"score": None, "epoch": epoch, "which": "live"}
            best_live_state = copy.deepcopy(model.state_dict())
            if averaged is not None:
                best_ema = {"score": None, "epoch": epoch, "which": "ema"}
                best_ema_state = copy.deepcopy(averaged.shadow.state_dict())
        else:
            if scored["live"]["score"] < best_live["score"]:
                best_live = {"score": scored["live"]["score"], "epoch": epoch, "which": "live"}
                best_live_state = copy.deepcopy(model.state_dict())
            if "ema" in scored and scored["ema"]["score"] < best_ema["score"]:
                best_ema = {"score": scored["ema"]["score"], "epoch": epoch, "which": "ema"}
                best_ema_state = copy.deepcopy(averaged.shadow.state_dict())

        # Written every epoch rather than once at the end, so a run in progress
        # can be watched -- or plotted -- from outside the process, and so an
        # interrupted run still leaves behind everything it learned. The "val"
        # key keeps its name whatever was scored; `scored_on` says what it was.
        write_json(out_dir / "history.json",
                   {"best": best, "epochs": history,
                    **({"best_ema": best_ema} if best_ema_state is not None else {}),
                    **({"best_live": best_live} if best_live_state is not None else {}),
                    **({"scored_on": label} if label != "val" else {})})

        # Every `checkpoint_every` epochs the run becomes recoverable: live
        # weights, moving average and optimiser state, written atomically. A
        # session that dies after this point loses at most that many epochs
        # instead of everything.
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            write_checkpoint(out_dir / "checkpoint.pt", cfg, model, averaged,
                             optimizer, epoch, step, best)

        if verbose:
            print(f"\r  epoch {epoch:3d}  {record['seconds']:5.1f}s  "
                  f"loss {train_metrics['total']:.4f}  "
                  f"ave {train_metrics['ave']:.4f}  "
                  f"platform_acc {train_metrics['platform_accuracy']:.3f}  "
                  + (f"grav {train_metrics['gravity_angle_deg']:.1f}deg  "
                     if gravity_on else "")
                  + "|  "
                  + "  ".join(f"{label}[{n}] {r['score']:.4f}"
                              + (f" grav {r['gravity_angle_deg']:.1f}deg"
                                 if r.get("gravity_angle_deg") is not None else "")
                              for n, r in scored.items())
                  + (f"lr {record['lr']:.2e}  (nothing held out)" if evaluate is None
                     else f"   <- best" if best["epoch"] == epoch else ""))

        if evaluate is not None and epoch - best["epoch"] >= patience:
            if verbose:
                print(f"  no improvement for {patience} epochs; stopping early")
            break

    if checkpoint_every:
        # The last one, written before `best_state` is loaded back in, so the
        # file left behind really is where training ended -- the live weights of
        # the final epoch, not the best epoch's. That is what a continuation
        # needs; the best epoch is already saved beside it under its own name.
        write_checkpoint(out_dir / "checkpoint.pt", cfg, model, averaged,
                         optimizer, len(history) - 1, step, best)

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": dict(cfg),
                "best": best, "n_parameters": model.n_parameters()},
               out_dir / "model.pt")
    # The two models, always: one per set of weights, so a notebook can predict
    # from each without asking which one won. One of them is the same state as
    # model.pt.
    for tag, state, info in (("live", best_live_state, best_live),
                             ("ema", best_ema_state, best_ema)):
        if state is not None:
            torch.save({"state_dict": state, "config": dict(cfg),
                        "best": info, "n_parameters": model.n_parameters()},
                       out_dir / f"model_{tag}.pt")
    if verbose:
        print(f"\nkept the final {best['which']} weights, epoch {best['epoch']}"
              if best["score"] is None else
              f"\nbest {label} score {best['score']:.4f} at epoch {best['epoch']} "
              f"({best['which']} weights)")
        for tag, info in (("live", best_live), ("ema", best_ema)):
            if info["epoch"] >= 0:
                print(f"  model_{tag}.pt: {tag} weights, epoch {info['epoch']}"
                      + (f", {label} {info['score']:.4f}" if info["score"] is not None
                         else " (final)"))
    return {"model": model, "best": best, "history": history}


def _plan_batches(cfg, dataset, epoch: int) -> list[list[int]]:
    """Batch groupings for one epoch, reshuffled by seeding on the epoch number.

    The seed is `seed_data * 1000 + epoch`, not `seed_data + epoch`. With the
    latter, a run seeded 43 would see at its first epoch exactly the ordering a
    run seeded 42 saw at its second -- the three seeds would be one shifted
    sequence rather than three independent ones, and the seed spread that M4
    measures would understate the real run-to-run variation. Multiplying keeps
    the streams apart.
    """
    return build_batches(
        dataset, int(cfg["data"]["batch_size"]), cfg["sampler"]["mode"],
        seed=int(cfg["run"]["seed_data"]) * 1000 + epoch)


# ------------------------------------------------------------------- score helper

def score_table(table, split: str, traj_ids=None) -> dict:
    """Score a prediction table, optionally restricted to some trajectories.

    Restriction is what makes the sealed holdout scoreable: it lives inside the
    *train* split, so scoring it means scoring a named subset of that split
    through the same code path val goes through.
    """
    return score_submission(table, split, traj_ids=traj_ids)
