"""Shared helpers: seeding, environment capture, run directories, result logging.

Tier-neutral. Nothing here knows about models, data, or the metric.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .paths import RESULTS_JSONL, ROOT, RUNS


# --------------------------------------------------------------------------- seeding

def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch from one integer.

    PipelinePlan.md §10 wants three *separate* seeds (weights, data order,
    augmentation), all recorded. This seeds one stream; the config carries the
    three values and each consumer calls this with its own.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- env

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def env_record() -> dict:
    """Everything needed to tell two runs apart that were not meant to differ."""
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "torch", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "absent"
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "device_available": device,
        "packages": versions,
    }


# --------------------------------------------------------------------------- json

def _jsonable(obj: Any) -> Any:
    """Make numpy scalars and Paths survive json.dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: str | Path, obj: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable) + "\n")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, obj: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_jsonable) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- hashing

def stable_hash(obj: Any, n: int = 8) -> str:
    """Deterministic short hash of a JSON-able object.

    `sort_keys` is what makes it stable across dict insertion orders, and
    `default=_jsonable` keeps numpy-typed config values from changing the hash
    depending on how they were parsed.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


# --------------------------------------------------------------------------- runs

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_dir(name: str, config_hash: str) -> Path:
    """runs/<YYYYMMDD>-<name>-<confighash8>/ per PipelinePlan.md §10."""
    d = RUNS / f"{utc_stamp()[:8]}-{name}-{config_hash}"
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    return d


def config_hash_appears(config_hash: str) -> bool:
    """Duplicate detection: has a run with this exact science already happened?"""
    return any(r.get("config_hash") == config_hash for r in read_jsonl(RESULTS_JSONL))


class Timer:
    """Wall-clock, because every results.jsonl line records one."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False
