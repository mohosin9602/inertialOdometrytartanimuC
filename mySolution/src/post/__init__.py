"""Tier 2 — runs on predictions.

Hard rule (PipelinePlan.md §3): **Tier 2 never reads a label.** Checked by
tests/test_m0.py.

`# noqa: F401` on the re-exports below tells a linter that "imported but unused"
"Yes, I know this import isn't directly used here. Don't complain."
This file exists to give the package a flat public surface.
"""
from .chain import OPS, PredictionAccumulator, run_chain  # noqa: F401
from .submission import write_submission  # noqa: F401
