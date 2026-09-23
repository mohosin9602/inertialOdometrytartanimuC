"""Tier 0 — offline, deterministic, cached preprocessing.

Hard rule (PipelinePlan.md §3): **Tier 0 never imports a model.** Checked by
tests/test_m0.py.
"""
from .cache import SplitCache, load_cache, cache_exists  # noqa: F401
