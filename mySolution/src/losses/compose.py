"""Loss composition: an ordered list of terms with absolute weights.

    score = 0.8156180650 * macro_AVE[m/s] + 0.1283685625 * macro_ATE20[m]

**Use the absolute weights, not normalised ones.** Then the loss *is* the score
up to an additive constant, gradients mirror the metric by construction, and a
training curve reads directly against the 0.4185 bar and the 0.0128 floor with
no conversion.

**The auxiliary terms break that property, so they are logged separately.**
Adding `0.1 * cross_entropy` (platform) or `0.1 * (1 - cos)` (gravity, M15) to
the total means the number on screen is no longer
comparable to a score. The bundle therefore reports two numbers: `score_proxy`,
which contains only the terms whose weights come from the metric, and `total`,
which is what the optimiser actually minimises. Two scalars in the log, not one.

One honest caveat about what any of this reproduces. The metric averages
hierarchically -- window, then trajectory, then platform, then equally over four
platforms. A batch-level mean reproduces the *innermost* average only. Matching
the outer three is `data.Sampler`'s job, not the loss's; the weights make the
terms commensurate, and the batch composition supplies the platform balance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from ..metric_constants import W_ATE_ABS, W_AVE_ABS
from .ate20 import ate20_from_batch
from .ave import masked_ave
from .gravity import gravity_angle_deg, gravity_cosine_loss
from .platform import platform_accuracy, platform_cross_entropy

TERMS = ("ave", "ate20", "platform", "gravity")

#: Terms whose weight comes from the competition metric. Only these are summed
#: into `score_proxy`, the number that can be read against 0.4185 directly.
METRIC_TERMS = ("ave", "ate20")


@dataclass
class LossBundle:
    """Ordered terms with absolute weights; returns the total and its parts."""

    terms: list[dict] = field(default_factory=list)

    def __call__(self, outputs: Mapping, batch: Mapping) -> tuple[torch.Tensor, dict]:
        v_pred, v_gt, mask = outputs["velocity"], batch["v_gt"], batch["mask"]
        total = v_pred.new_zeros(())
        score_proxy = v_pred.new_zeros(())
        parts: dict[str, float] = {}

        for term in self.terms:
            if not term.get("enabled", True):
                continue
            name, w = term["name"], float(term["weight"])

            if name == "ave":
                value = masked_ave(v_pred, v_gt, mask)
            elif name == "ate20":
                # The metric itself, not a proxy for it: the scorer rotates
                # body-frame velocity with the same ground-truth quaternions the
                # training data ships. Runs on CPU in float64 whatever device the
                # model is on -- see losses/ate20.py for why that is faster, not
                # slower, and for the three traps inside it.
                value = ate20_from_batch(v_pred, batch)
            elif name == "platform":
                if "platform_logits" not in outputs:
                    raise KeyError(
                        "the platform loss term is enabled but the model returned no "
                        "'platform_logits'. Either switch the term off or build the "
                        "model with heads.platform = true.")
                value = platform_cross_entropy(outputs["platform_logits"],
                                               batch["platform_id"])
                parts["platform_accuracy"] = platform_accuracy(
                    outputs["platform_logits"], batch["platform_id"])
            elif name == "gravity":
                # Auxiliary, like the platform term, and for the same reason it
                # stays out of METRIC_TERMS: it lands in `total`, never in
                # `score_proxy`. See losses/gravity.py.
                if "gravity" not in outputs:
                    raise KeyError(
                        "the gravity loss term is enabled but the model returned no "
                        "'gravity'. Either switch the term off or build the model "
                        "with heads.gravity = true.")
                value = gravity_cosine_loss(outputs["gravity"], batch["q_gt"],
                                            mask)
                parts["gravity_angle_deg"] = gravity_angle_deg(
                    outputs["gravity"], batch["q_gt"], mask)
            else:
                raise KeyError(f"unknown loss term {name!r}; have {list(TERMS)}")

            parts[name] = float(value.detach())
            total = total + w * value
            if name in METRIC_TERMS:
                score_proxy = score_proxy + w * value

        parts["score_proxy"] = float(score_proxy.detach())
        parts["total"] = float(total.detach())
        return total, parts


def build_loss(cfg: Mapping) -> LossBundle:
    return LossBundle(terms=[dict(t) for t in cfg["loss"]["terms"]])


def term_enabled(cfg: Mapping, name: str) -> bool:
    """Is loss term `name` switched on in this config?

    The dataloader needs to know: the ATE20 term reads segment bounds that only
    get into the batch if the segment cache was loaded, and a term silently
    training on zero segments would look exactly like a term that does not help.
    """
    return any(t["name"] == name and t.get("enabled", True)
               for t in cfg["loss"]["terms"])


#: Re-exported so a caller never has to retype them.
ABSOLUTE_WEIGHTS = {"ave": W_AVE_ABS, "ate20": W_ATE_ABS}
