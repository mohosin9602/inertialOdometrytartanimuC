"""Tier 1 — the forward pass.

Hard rule (PipelinePlan.md §3): **Tier 1 never reads a file.** Everything a
model needs arrives in the batch. Checked by tests/test_m0.py.
"""
from .base import MODEL_REGISTRY, VelocityModel, build_model, forward_batch  # noqa: F401
from .conditioner import build_conditioner  # noqa: F401
from .context import build_context  # noqa: F401
from .embodiment import Embodiment  # noqa: F401
from .encoder import build_encoder  # noqa: F401
from .heads import GravityHead, PlatformHead, VelocityHead  # noqa: F401
from .masked import masked_mean, zero_padding  # noqa: F401
from .two_stage import TwoStageModel  # noqa: F401
from .zero import ConstantZeroModel  # noqa: F401
