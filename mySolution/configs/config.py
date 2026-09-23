"""Superseded — the live configuration system is `mySolution/src/config.py`.

This file was an early placeholder holding a handful of loose constants. Keeping
two config sources is how ablation tables drift, so nothing imports it any more:

* **Paths** live in `src/paths.py`, derived from the source file's own location.
* **Scoring constants** live in `src/metric_constants.py`, read from the vendored
  scorer rather than retyped, and asserted against the published literals.
* **Run configuration** lives in `src/config.py`: one resolved, hashed object per
  run, expressed as a diff against a named, versioned reference config
  (PipelinePlan.md §5).

Left in place only so an old import fails loudly with this explanation instead of
silently picking up stale values.
"""
raise ImportError(__doc__)
