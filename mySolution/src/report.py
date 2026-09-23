"""Turn `runs/results.jsonl` into the tables the technical report needs.

    mySolution/.venv/bin/python -m src.report                     # every run
    mySolution/.venv/bin/python -m src.report --milestone M5
    mySolution/.venv/bin/python -m src.report --spread m5-seed    # the noise floor
    mySolution/.venv/bin/python -m src.report --bakeoff model.encoder.mode

**Generate the tables, never assemble them by hand.** That rule is in models.md
§10 for a reason: an ablation table typed out from notes is a table nobody can
re-derive three weeks later, and by then the run directories are the only place
the truth still lives. Every number this prints comes straight out of the
append-only results log.

**The `--spread` view is milestone M4**, the noise floor. It reports the mean
and the spread of one configuration run at several seeds. That spread is the
number every later claim is measured against: an improvement smaller than it is
not an improvement, it is the same run twice.

**The name prefix is a starting point, not the definition of the group.** A
spread is only meaningful across runs that are the same configuration apart
from the seed, and run names do not guarantee that -- `m8-bigru-60ep-seed42`
(named `m8-bigru-seed42` at the time) was a 60-epoch run while
`m8-bigru-seed43` was a 150-epoch one, and averaging them reported a standard
deviation nine times the truth. So the prefix picks a
starting run, its **resolved configuration** defines the arm, and every run in
the log sharing that configuration joins the group no matter what it is called.
Runs that match the name but not the configuration are excluded and named.

**The `--bakeoff` view is every rung of the ladder from M6 onward.** It groups
runs by what one config key was set to and prints them **drone-first**, because
drone is 3 to 6 times worse than every other platform and therefore holds
essentially all of the remaining score (STATUS.md). Reading a bake-off by its
aggregate score alone hides the only quarter of the metric that is still moving.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Mapping

import numpy as np

from .config import REFERENCES, deep_merge, diff_against
from .metric_constants import (VAL_BASELINE_FORCED, VAL_BASELINE_ROUTED,
                               VAL_FLOOR, VAL_ZEROS)
from .paths import PLATFORMS, RESULTS_JSONL
from .utils import read_jsonl, stable_hash

#: Drone first. The metric weights the four platforms equally, so a table read
#: in alphabetical order buries the platform that is costing the points.
DRONE_FIRST = ("drone",) + tuple(p for p in PLATFORMS if p != "drone")


# ------------------------------------------------- reconstructing what ran

def _nest(flat: Mapping[str, Any]) -> dict:
    """Turn a flat dotted-key config diff back into the nested shape it came from."""
    out: dict = {}
    for dotted, value in flat.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


_RESOLVED: dict[str, dict] = {}


def resolved_config(row: dict) -> dict:
    """The full configuration a logged run actually ran, rebuilt from the log line.

    A results.jsonl line stores only `reference_name` plus the diff, so two runs
    of the *same* configuration reached from different references look different
    on the page -- `m8long-bigru-seed42` is `m5_two_stage` plus eight overrides
    while `m8-bigru-seed43` is `m8_bigru` with an empty diff, and they are the
    same run at two seeds. Comparing resolved configurations instead of diffs is
    what lets a promotion rebase the reference without severing every earlier arm
    from every later one. That matters at every rung from here on, because each
    promotion splits the log's lineage again.

    This reconstruction is only sound because references are **versioned** and
    every row records which version it used. If a reference were ever edited in
    place rather than promoted to a new version, every historical row would
    silently rebuild into a configuration that never ran -- so the version is
    checked here and a mismatch is a hard error, never a warning.
    """
    name = row.get("reference_name") or ""
    if name not in REFERENCES:
        raise KeyError(
            f"run {row.get('run_name')!r} names reference {name!r}, which is not in "
            f"config.REFERENCES (have {sorted(REFERENCES)}). Its configuration "
            f"cannot be rebuilt, so it cannot be compared with anything.")
    ref = REFERENCES[name]
    logged = row.get("reference_version")
    if logged is not None and logged != ref["version"]:
        raise ValueError(
            f"run {row.get('run_name')!r} ran against {name} v{logged} but "
            f"config.py now holds v{ref['version']}. A reference was edited in "
            f"place instead of being promoted to a new version, so this run's "
            f"configuration can no longer be rebuilt. Restore the old version "
            f"under its own name before reading any table.")
    memo = json.dumps([name, logged, row.get("config_diff") or {}], sort_keys=True)
    if memo not in _RESOLVED:
        _RESOLVED[memo] = deep_merge(ref["config"], _nest(row.get("config_diff") or {}))
    return _RESOLVED[memo]


def _flat(cfg: Mapping, prefix: str = "") -> dict[str, Any]:
    """A nested config as flat dotted keys, for comparing two runs key by key."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, Mapping):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_runs(milestone: str | None = None, name_prefix: str | None = None,
              scored_only: bool = True) -> list[dict]:
    rows = read_jsonl(RESULTS_JSONL)
    if scored_only:
        rows = [r for r in rows if r.get("metrics")]
    if milestone:
        rows = [r for r in rows if r.get("milestone") == milestone]
    if name_prefix:
        rows = [r for r in rows if str(r.get("run_name", "")).startswith(name_prefix)]
    return rows


def runs_table(rows: list[dict]) -> str:
    """One line per run: what changed, what it scored, and on which platforms."""
    header = (f"{'run':<20} {'ms':<9} {'score':>7} {'seal':>7} "
              + " ".join(f"{p:>7}" for p in PLATFORMS) + "  diff")
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: r["timestamp"]):
        m = r["metrics"]
        per = m.get("per_platform", {})
        seal = (r.get("seal") or {}).get("score")
        diff = ", ".join(f"{k}={v}" for k, v in sorted((r.get("config_diff") or {}).items())
                         if not k.startswith("run."))
        lines.append(
            f"{r['run_name']:<20} {str(r.get('milestone','')):<9} {m['score']:>7.4f} "
            + (f"{seal:>7.4f} " if seal is not None else f"{'-':>7} ")
            + " ".join(f"{per.get(p, {}).get('ave', float('nan')):>7.3f}"
                       for p in PLATFORMS)
            + f"  {diff or '(reference)'}")
    return "\n".join(lines)


def spread_table(rows: list[dict], prefix: str) -> str:
    """Mean, spread and range of one configuration at several seeds -- the M4 floor.

    Two spreads are reported and they answer different questions. The standard
    deviation says how much a score wobbles run to run; the full range says how
    unlucky a single run can be, which is what matters when a later ablation is
    judged from one run rather than three.

    `rows` is the whole log, not a pre-filtered group. `prefix` picks the run
    that names the arm; the arm itself is then defined by that run's resolved
    configuration, so every seed of it is found however it happens to be named,
    and nothing else can slip in. Grouping by name alone once averaged a
    60-epoch run together with two 150-epoch ones and reported a standard
    deviation of 0.0147 where the truth was 0.0016 -- an inflated floor is the
    most expensive kind of reporting bug, because it silently retires real
    improvements as noise.
    """
    matched = [r for r in rows if str(r.get("run_name", "")).startswith(prefix)]
    label = f"seed spread for {prefix!r}"
    if not matched:
        return f"{label}: no run in {RESULTS_JSONL} has a name starting with {prefix!r}."

    # The arm is whichever configuration most of the matching runs share; every
    # run in the log with that configuration then joins it, named or not.
    by_arm: dict[str, list[dict]] = {}
    for r in matched:
        by_arm.setdefault(arm_signature(r), []).append(r)
    signature = max(by_arm,
                    key=lambda k: (len(by_arm[k]), by_arm[k][-1].get("timestamp", "")))
    group = sorted((r for r in rows if arm_signature(r) == signature),
                   key=lambda r: r.get("timestamp", ""))
    excluded = [r for r in matched if arm_signature(r) != signature]
    adopted = [r for r in group if r not in matched]

    notes: list[str] = []
    if excluded:
        rep = _flat(resolved_config(group[0]))
        notes += ["  EXCLUDED -- these match the name but are a different configuration,",
                  "  so averaging them in would report a design change as seed noise:"]
        for r in excluded:
            other = _flat(resolved_config(r))
            differs = {k: (rep.get(k), other.get(k)) for k in set(rep) | set(other)
                       if k not in _IDENTITY_KEYS and rep.get(k) != other.get(k)}
            shown = ", ".join(f"{k} {b!r} not {a!r}" for k, (a, b) in sorted(differs.items()))
            notes.append(f"    {r.get('run_name')}: {shown or 'differs outside the config'}")
        notes.append("")
    if adopted:
        notes += ["  ALSO INCLUDED -- same configuration, different name:",
                  "    " + ", ".join(str(r.get("run_name")) for r in adopted), ""]

    rows = group
    if len(rows) < 2:
        return "\n".join(notes + [f"{label}: only {len(rows)} run(s) share this "
                                  "configuration; a spread needs at least two."])

    def summarise(name: str, values: list[float]) -> str:
        v = np.asarray(values, float)
        return (f"  {name:<14} mean {v.mean():.4f}   sd {v.std(ddof=1):.4f}   "
                f"range {v.min():.4f} to {v.max():.4f}   "
                f"(spread {v.max() - v.min():.4f})")

    lines = notes + [f"{label}: {len(rows)} runs -- "
                     + ", ".join(str(r.get("run_name")) for r in rows), ""]
    lines.append(summarise("val score", [r["metrics"]["score"] for r in rows]))
    lines.append(summarise("val AVE", [r["metrics"]["macro_ave"] for r in rows]))
    lines.append(summarise("val ATE20", [r["metrics"]["macro_ate20"] for r in rows]))
    seals = [r["seal"]["score"] for r in rows if r.get("seal")]
    if len(seals) >= 2:
        lines.append(summarise("seal score", seals))

    lines += ["", "  per platform (val AVE, m/s):"]
    for p in DRONE_FIRST:
        vals = [r["metrics"]["per_platform"][p]["ave"] for r in rows
                if p in r["metrics"].get("per_platform", {})]
        if len(vals) >= 2:
            v = np.asarray(vals, float)
            lines.append(f"    {p:<7} mean {v.mean():.4f}   sd {v.std(ddof=1):.4f}")

    scores = np.asarray([r["metrics"]["score"] for r in rows], float)
    sd = float(scores.std(ddof=1))
    lines += ["", f"  READ THIS AS: a later change worth less than about "
                  f"{2 * sd:.4f} score points", "  (two standard deviations) is "
                  "not distinguishable from running the same config twice."]
    return "\n".join(lines)


# ------------------------------------------------------------------- bake-offs

#: Keys that describe *which* run this is rather than *what* it tested. Two arms
#: are still comparable when these differ; anything else differing means the two
#: runs changed more than one thing and do not belong in the same table.
#: `eval.seal` is here on purpose. Scoring the sealed holdout happens *after*
#: training, on the already-fitted model, and changes neither the training nor
#: the val score -- so a run that opened the seal and one that did not are still
#: measuring the same thing on val. The seal is a scarce resource with a much
#: larger noise floor than val (0.0155 against 0.0037), so it cannot arbitrate a
#: bake-off anyway; most arms leave it shut and the incumbent's seal number still
#: shows in the table.
_IDENTITY_KEYS = ("run.name", "run.notes", "run.milestone",
                  "run.seed_weights", "run.seed_data", "run.seed_augment",
                  "eval.seal")


def resolve_key(row: dict, key: str):
    """What `key` was actually set to in this run, diff or reference default alike.

    Read straight out of the rebuilt configuration, so the incumbent arm of a
    bake-off sits in the table beside its challengers without having to be
    re-run: `cnn_small` is what the reference already says, so the three M5 runs
    changed nothing and their diff is empty, yet the value is still there.
    """
    node: Any = resolved_config(row)
    for part in key.split("."):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]        # `loss.terms.1.enabled` and friends
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _step(node, part):
    """One dotted-path step into a dict or into a list by index."""
    if isinstance(node, list) and part.isdigit() and int(part) < len(node):
        return node[int(part)]
    return node.get(part) if isinstance(node, dict) else None


def _without(cfg: Mapping, dotted_keys) -> dict:
    """A copy of `cfg` with each dotted key removed, missing ones ignored."""
    out = copy.deepcopy(dict(cfg))
    for dotted in dotted_keys:
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = _step(node, part)
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return out


def arm_signature(row: dict, key: str | None = None) -> str:
    """A fingerprint of everything this run held fixed apart from the key under test.

    Two runs belong in the same table only when this matches. It is what keeps a
    one-epoch smoke test, a CPU run, or a different epoch budget out of a
    comparison they would silently distort.

    **Fingerprinting the resolved configuration, not the diff, is the point.**
    The diff is relative to a reference that moves every time a bake-off promotes
    a winner, so two identical runs straddling a promotion carry different
    diffs -- and a table keyed on the diff would split them apart while claiming
    to compare like with like. The resolved configuration is what the run
    actually was, and it does not care how it was reached.
    """
    stripped = _without(resolved_config(row),
                        _IDENTITY_KEYS + ((key,) if key else ()))
    return stable_hash(stripped, 16)


def _holding_constant(row: dict, key: str | None = None) -> str:
    """How this arm differs from its own reference -- the header line of a table."""
    ref = REFERENCES[row["reference_name"]]["config"]
    diff = diff_against(ref, resolved_config(row))
    shown = {k: v for k, v in diff.items()
             if k != key and k not in _IDENTITY_KEYS}
    return f"{row['reference_name']}" + (f" + {shown}" if shown else " (unchanged)")


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    v = np.asarray(values, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)


def _epoch_note(group: list[dict]) -> str:
    """`best/run`, with a `*` when the epoch budget ran out before early stopping.

    A fairness check the bake-off gets for free. Arms differ in size by a factor
    of eight here, and a larger model that had not finished learning when the
    budget expired would score badly for a reason that has nothing to do with the
    architecture. Early stopping firing is the signal that a run had genuinely
    finished; using the whole budget instead is the signal that it had not. If a
    starred arm loses, the honest conclusion is "not better at this budget"
    rather than "not better".
    """
    best = [r.get("best_epoch") for r in group if r.get("best_epoch") is not None]
    ran = [r.get("epochs_run") for r in group if r.get("epochs_run")]
    if not best or not ran:
        return "-"
    b, n = int(np.mean(best)), int(np.mean(ran))
    budget = resolve_key(group[0], "optim.epochs")
    capped = isinstance(budget, int) and n >= budget
    return f"{b}/{n}" + ("*" if capped else "")


def bakeoff_table(rows: list[dict], key: str) -> str:
    """Compare the arms of one ablation rung, read drone-first.

    Arms are grouped by what `key` was set to, and only arms that changed nothing
    else are shown. The score column decides the milestone; the drone column is
    printed first and discussed underneath, because drone alone contributes about
    0.20 of the current 0.3663 and halving it is worth more than any other lever
    left on the board.
    """
    families: dict[str, list[dict]] = {}
    for r in rows:
        if resolve_key(r, key) is None:
            continue
        families.setdefault(arm_signature(r, key), []).append(r)
    if not families:
        return f"no runs in {RESULTS_JSONL} set {key!r} to anything."

    # The family worth printing is the one with the most distinct arms; ties go
    # to the one with the most runs behind it.
    def rank(group: list[dict]) -> tuple[int, int]:
        return len({str(resolve_key(r, key)) for r in group}), len(group)

    _signature, family = max(families.items(), key=lambda kv: rank(kv[1]))
    rep = max(family, key=lambda r: r.get("timestamp", ""))

    arms: dict[str, list[dict]] = {}
    for r in family:
        arms.setdefault(str(resolve_key(r, key)), []).append(r)
    ordered = sorted(arms.items(),
                     key=lambda kv: _mean_sd([x["metrics"]["score"] for x in kv[1]])[0])

    # The noise floor, measured rather than typed: the widest per-arm spread in
    # this very table. An arm with one run has no spread of its own, so a
    # single-run arm is judged against whichever arm was repeated.
    seed_sds = [sd for _, group in arms.items()
                if (sd := _mean_sd([r["metrics"]["score"] for r in group])[1]) is not None]
    floor = 2 * max(seed_sds) if seed_sds else None
    drone_sds = [sd for _, group in arms.items()
                 if (sd := _mean_sd([r["metrics"]["per_platform"]["drone"]["ave"]
                                     for r in group])[1]) is not None]
    drone_floor = 2 * max(drone_sds) if drone_sds else None

    best_name, best_group = ordered[0]
    best_score = _mean_sd([r["metrics"]["score"] for r in best_group])[0]

    head = (f"{'arm':<12} {'n':>2} {'score':>8} {'sd':>7} {'d.score':>8} "
            + " ".join(f"{p:>7}" for p in DRONE_FIRST)
            + f" {'seal':>7} {'params':>10} {'min':>6} {'best ep':>8}")
    lines = [f"bake-off on {key}   ({len(family)} runs, {len(arms)} arms)",
             f"holding constant: {_holding_constant(rep, key)}",
             "", head, "-" * len(head)]
    for name, group in ordered:
        score, sd = _mean_sd([r["metrics"]["score"] for r in group])
        seal_vals = [r["seal"]["score"] for r in group if r.get("seal")]
        per = {p: _mean_sd([r["metrics"]["per_platform"][p]["ave"] for r in group])[0]
               for p in DRONE_FIRST if p in group[0]["metrics"]["per_platform"]}
        lines.append(
            f"{name:<12} {len(group):>2} {score:>8.4f} "
            + (f"{sd:>7.4f} " if sd is not None else f"{'-':>7} ")
            + f"{score - best_score:>+8.4f} "
            + " ".join(f"{per.get(p, float('nan')):>7.3f}" for p in DRONE_FIRST)
            + (f" {np.mean(seal_vals):>7.4f}" if seal_vals else f" {'-':>7}")
            + f" {group[0].get('n_parameters') or 0:>10,}"
            + f" {np.mean([r['wall_clock_s'] for r in group]) / 60:>6.0f}"
            + f" {_epoch_note(group):>8}")

    # Drone-first reading, spelled out rather than left to the reader.
    drone_rank = sorted(
        ((name, _mean_sd([r["metrics"]["per_platform"]["drone"]["ave"] for r in g])[0])
         for name, g in arms.items() if "drone" in g[0]["metrics"]["per_platform"]),
        key=lambda kv: kv[1])
    if any(_epoch_note(g).endswith("*") for g in arms.values()):
        lines += ["", "  * this arm used its whole epoch budget -- early stopping "
                      "never fired, so it had not",
                  "    finished learning. A loss here means 'not better at this "
                  "budget', not 'not better'."]
    lines += ["", "  DRONE FIRST. Drone is 3 to 6x worse than every other platform "
                  "and holds most of the",
              "  remaining score, so it is the column to read before the aggregate."]
    if len(drone_rank) > 1:
        (d_best, d_bv), (d_worst, d_wv) = drone_rank[0], drone_rank[-1]
        lines.append(f"    best drone AVE  {d_best} at {d_bv:.3f} m/s   "
                     f"(worst: {d_worst} at {d_wv:.3f})")
        if drone_floor is not None:
            verdict = ("real" if (d_wv - d_bv) > drone_floor
                       else "INSIDE the seed noise -- not a result")
            lines.append(f"    spread across arms {d_wv - d_bv:.3f} m/s against a "
                         f"2-sigma seed floor of {drone_floor:.3f}  ->  {verdict}")
    if floor is not None and len(ordered) > 1:
        runner, runner_group = ordered[1]
        gap = _mean_sd([r["metrics"]["score"] for r in runner_group])[0] - best_score
        verdict = ("a real win" if gap > floor
                   else "INSIDE the seed noise -- the arms are tied")
        lines += ["", f"  ON SCORE: {best_name} leads {runner} by {gap:.4f} against a "
                      f"2-sigma seed floor of {floor:.4f}  ->  {verdict}"]
    elif len(ordered) == 1:
        lines += ["", f"  Only one arm has been run. Run the others with:",
                  f"    .venv/bin/python -m src.bakeoff --key {key} "
                  f"--values ... --tag ... --milestone ..."]
    return "\n".join(lines)


def reference_points() -> str:
    return ("reference points on val\n"
            f"  {VAL_FLOOR:.4f}  ground-truth velocities, the floor\n"
            f"  {VAL_BASELINE_ROUTED:.4f}  released baseline routed by true platform "
            f"-- a TARGET, illegal to submit\n"
            f"  {VAL_BASELINE_FORCED:.4f}  released baseline with no routing "
            f"-- the bar a legal model must beat\n"
            f"  {VAL_ZEROS:.4f}  all zeros")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--milestone", default=None)
    ap.add_argument("--spread", default=None, metavar="NAME_PREFIX",
                    help="report the seed spread of the configuration named by "
                         "this prefix (every seed of it, however named)")
    ap.add_argument("--bakeoff", default=None, metavar="DOTTED_KEY",
                    help="compare the arms of one ablation rung, drone-first")
    a = ap.parse_args()

    print(reference_points())
    print()
    if a.spread:
        print(spread_table(load_runs(), a.spread))
        print()
    if a.bakeoff:
        print(bakeoff_table(load_runs(), a.bakeoff))
        print()
    print(runs_table(load_runs(a.milestone)))


if __name__ == "__main__":
    main()
