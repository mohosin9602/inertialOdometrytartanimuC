"""mySolution — TartanIMU Challenge pipeline.

Tier layout, per .claude/PipelinePlan.md §3:

    prep/        Tier 0    offline, deterministic, cached. Never imports a model.
    data/        Tier 0    dataset / batching. Produces the §2.5 batch object.
    models/      Tier 1    the forward pass. Never reads a file.
    losses/      Tier 1    scalar objectives. May read labels.
    post/        Tier 2    runs on predictions. Never reads a label.
    evaluation/  Tier 2    scoring. Named `evaluation` rather than the plan's `eval`
                           only to avoid shadowing the builtin in local namespaces.

The tier rules are checked by tests/test_m0.py, not left to discipline.
"""
