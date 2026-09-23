"""Run one ablation rung: the same configuration N times, one line changed each time.

    mySolution/.venv/bin/python -m src.bakeoff --key model.encoder.mode \
        --values mlp resnet1d tcn --tag m6 --milestone M6

That is milestone M6, the backbone bake-off. The same command shape runs every
later rung of the ladder in models.md §10 -- M7 moves `data.input_repr`, M8 moves
`model.context.mode`, M9 moves `model.conditioner.mode` -- which is why this is a
general runner rather than four copies of a script.

**What it does and does not do.** It resolves the reference config, overrides one
dotted key per arm, and calls the ordinary `pipeline.run`. Nothing about training,
scoring or logging is special-cased here: every arm lands in `runs/results.jsonl`
exactly like a hand-run experiment, and the comparison afterwards is read back out
of that file by `src.report`. If this script disappeared, the same runs could be
reproduced by typing four `--set` commands.

**It is resumable, and that matters.** A four-arm encoder bake-off is close to
three hours of GPU time on this laptop. Each arm is skipped when its config hash
is already in the results log, so an interrupted bake-off continues where it
stopped rather than starting again. That is the same duplicate-detection rule
`pipeline.run` enforces; this only checks it early enough to skip politely
instead of raising.

**The arm you are comparing against usually already exists.** The incumbent needs
no re-run: `cnn_small` was measured three times at M5, so M6 only has to run the
three challengers. Pass just the values that are new.
"""
from __future__ import annotations

import argparse
import json
import traceback

from . import config as config_mod
from . import pipeline
from .paths import RESULTS_JSONL
from .report import bakeoff_table, load_runs
from .utils import config_hash_appears


def arm_overrides(key: str, value, tag: str, milestone: str, seed: int) -> dict:
    """The override dict for one arm: the key under test, plus its identity.

    The run *name* carries the arm so that `results.jsonl` reads plainly, but the
    name is deliberately outside the config hash -- two arms are different runs
    because their science differs, not because they were labelled differently.
    """
    over: dict = {"run": {"name": f"{tag}-{value}-seed{seed}",
                          "notes": f"{milestone} bake-off arm: {key} = {value}",
                          "milestone": milestone,
                          "seed_weights": seed, "seed_data": seed,
                          "seed_augment": seed}}
    node = over
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return over


def run_bakeoff(key: str, values: list, tag: str, milestone: str,
                reference: str = "m5_two_stage", seed: int = 42,
                extra: dict | None = None, verbose: bool = True) -> list[dict]:
    """Run every arm that has not been run before, then return the records."""
    records = []
    for i, value in enumerate(values, 1):
        over = arm_overrides(key, value, tag, milestone, seed)
        if extra:
            over = config_mod.deep_merge(over, extra)
        cfg = config_mod.resolve(reference, over)
        head = f"[{i}/{len(values)}] {key} = {value}   #{cfg.hash}"
        print(f"\n{'=' * 78}\n{head}\n{'=' * 78}")
        if config_hash_appears(cfg.hash):
            print(f"already in {RESULTS_JSONL} -- skipping. "
                  f"Delete the line or change the config to run it again.")
            continue
        try:
            records.append(pipeline.run(reference, over, verbose=verbose))
        except Exception:
            # One arm failing must not throw away the arms that already ran; the
            # results log holds them, and the summary below still prints.
            print(f"ARM FAILED: {key} = {value}\n{traceback.format_exc()}")
    return records


def _parse_value(raw: str):
    """An arm value from the command line, JSON-typed, falling back to the string.

    **A boolean arm is the reason this exists.** `--values true false` arrives
    from argparse as two strings, and a non-empty string is truthy, so an arm
    meant to switch a loss term OFF would switch it on -- both arms would train
    identically, hash differently, and be reported as a clean comparison. Typing
    the value the same way `--set` does makes `true`/`false`, numbers and lists
    mean what they say, while a bare word like `resnet1d` stays a string.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--key", required=True,
                    help="dotted config key under test, e.g. model.encoder.mode")
    ap.add_argument("--values", nargs="+", required=True,
                    help="one arm per value; already-run arms are skipped")
    ap.add_argument("--tag", required=True, help="run-name prefix, e.g. m6")
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--reference", default="m5_two_stage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", action="append", dest="sets", default=[],
                    metavar="KEY=VALUE",
                    help="extra override applied to EVERY arm, so the comparison "
                         "stays like-for-like")
    a = ap.parse_args()

    extra = pipeline._parse_set(a.sets)
    run_bakeoff(a.key, [_parse_value(v) for v in a.values],
                a.tag, a.milestone, a.reference, a.seed, extra)

    print(f"\n{'=' * 78}\nBAKE-OFF SUMMARY: {a.key}\n{'=' * 78}")
    print(bakeoff_table(load_runs(), a.key))


if __name__ == "__main__":
    main()
