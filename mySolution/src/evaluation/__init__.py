"""Tier 2 — scoring. Wraps the vendored official scorer; never reimplements it."""
from .score import (build_solution, score_submission, format_report,  # noqa: F401
                    verify_solution_matches_repro)
