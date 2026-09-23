"""Loss terms, composed with ABSOLUTE metric weights. PipelinePlan.md §4.4."""
from .ate20 import (ate20_from_batch, masked_ate20,  # noqa: F401
                    segment_rms, umeyama_align)
from .ave import masked_ave, per_window_ave  # noqa: F401
from .compose import LossBundle, build_loss, term_enabled  # noqa: F401
from .gravity import (gravity_angle_deg, gravity_angles_deg,  # noqa: F401
                      gravity_cosine_loss, gravity_valid)
from .platform import platform_accuracy, platform_cross_entropy  # noqa: F401
