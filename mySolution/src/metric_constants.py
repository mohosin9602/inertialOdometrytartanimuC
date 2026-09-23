"""The competition's scoring constants, taken from the vendored scorer.

    TartanIMU Score = 0.6 * (macro_AVE / 0.7356384388) + 0.4 * (macro_ATE20 / 3.1160277267)

Folding the normalisers in gives the form a training loss should use
(PipelinePlan.md §4.4):

    score = 0.8156180650 * macro_AVE[m/s] + 0.1283685625 * macro_ATE20[m]

Use those **absolute** weights, never normalised ones. Then the loss is the
score up to an additive constant and a training curve reads directly against the
0.4185 bar and the 0.0128 floor with no conversion.

The values are read from the vendored scorer rather than retyped, and then
asserted against the literals above so a silent upstream change is caught here
instead of showing up as an unexplained shift in an ablation table.
"""
from __future__ import annotations

from .paths import add_repro_to_path

add_repro_to_path()
import kaggle_metric_tartanimu_score as _K  # noqa: E402 # "./TartanIMU/starter" is added to python paths. It lives there.

W_AVE = _K.W_AVE                # 0.6
W_ATE = _K.W_ATE                # 0.4
AVE_REF = _K.AVE_REF            # m/s,  all-zeros AVE on the full test set
ATE_REF = _K.ATE_REF            # m,    all-zeros ATE20 on the full test set

# W_AVE_ABS and W_ATE_ABS are the absolute weights for the loss function, which are derived from the normalized weights and reference values. They represent the contribution of the average velocity error (AVE) and average trajectory error (ATE) to the overall score in absolute terms.
W_AVE_ABS = W_AVE / AVE_REF     # 0.8156180650  (per m/s of AVE)
W_ATE_ABS = W_ATE / ATE_REF     # 0.1283685625  (per m of ATE20)
AVE_OVER_ATE = W_AVE_ABS / W_ATE_ABS   # 6.3537: 1 cm/s is worth 6.35 cm

# Reference points on the labelled val split, all [measured]; see CLAUDE.md.
VAL_FLOOR = 0.0128              # ground-truth velocities fed back through the scorer
VAL_ZEROS = 1.0144              # all-zeros submission  <- the M0 success criterion
VAL_BASELINE_ROUTED = 0.4185    # released baseline routed by true platform: the bar
VAL_BASELINE_FORCED = 0.8346    # released baseline with --head human forced

assert abs(W_AVE_ABS - 0.8156180650) < 1e-9, W_AVE_ABS
assert abs(W_ATE_ABS - 0.1283685625) < 1e-9, W_ATE_ABS
