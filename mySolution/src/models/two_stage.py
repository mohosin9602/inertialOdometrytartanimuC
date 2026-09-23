"""The M5 model: five stages wired together into one network. models.md §2.

    batch["x"]  (B, K, C, T)          B chunks, K windows, C channels, 200 frames
         |
         |  fold the chunk and window axes together
         v
    (1) WindowEncoder      (B*K, C, T) -> (B*K, D)      one summary per second
         |
         z  (B, K, D)
         |
         +--------------------------------+
         v                                v
    (2) TrajectoryContext            (3) Embodiment
        neighbours share information     masked average over the chunk,
        (identity at M5)                 then a small network -> e (B, E)
         |                                |
         +---> (5) GravityHead  (B, K, 3) +---> (5) PlatformHead  (B, 4)
         |     M15, off by default.       |         AUXILIARY LOSS ONLY.
         |     AUXILIARY LOSS ONLY.       |         Nothing below reads this.
         v                                v
    (4) Conditioner  <-------------------- e
        adjust the features using e
         |
         v
    (5) VelocityHead       (B, K, 3)      body-frame velocity, m/s

**The one hard rule this file enforces.** The velocity output must not depend on
any platform decision made outside the network, and it must not depend on the
auxiliary platform head either. Concretely: `forward` computes the velocity from
`x` and `mask` alone; the platform logits are produced afterwards and are never
fed back. Deleting the head entirely would change no prediction. That is not a
convention, it is the competition rule, and `tests/test_m5.py` tests all three
halves of it mechanically.

**Why the embodiment branch reads the encoder output rather than the context
output.** Two reasons. It keeps the M8 context comparison and the M9 conditioning
comparison independent, so changing one does not move the other underneath it.
And it is sufficient: the whole 100%-accurate platform-identification result was
built on hand-computed statistics of individual windows, and the encoder's
summaries are strictly richer than those.

**The `cond` argument is accepted and ignored.** The batch still carries a
`cond` slot from the old design, where an external classifier's output was fed in
as data. The 2026-08-30 rules clarification retired that. This model builds its
own embodiment vector internally, so `cond` reaches it and goes nowhere -- and
`tests/test_m5.py` proves that by scrambling `cond` and checking that not one
prediction moves.
"""
from __future__ import annotations

import torch

from .base import VelocityModel, register
from .conditioner import build_conditioner
from .context import build_context
from .embodiment import Embodiment
from .encoder import build_encoder
from .heads import GravityHead, PlatformHead, VelocityHead


#: Where the three gravity channels sit in every nine-channel representation.
GRAVITY_CHANNELS = slice(6, 9)
ATTITUDE_MODES = ("late", "both")


class AttitudeBranch(torch.nn.Module):
    """AirIO's separate attitude encoder (M21), for the gravity direction.

    AirIO (Qiu et al., RA-L 2025) feeds the body-frame IMU to one encoder and the
    attitude, as a 3-vector, to a second, and joins them before the recurrent
    stage. Here the attitude is the only part of it an IMU can observe, the
    gravity direction, summarised per window as its mean and its change across
    the window. The output layer starts at zero, so a model with the branch
    begins exactly where the model without it begins and has to learn any use of
    it -- the same device FiLM uses (models.md).
    """

    def __init__(self, d_model: int, width: int = 64):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(6, width), torch.nn.GELU(),
                                       torch.nn.Linear(width, width), torch.nn.GELU())
        self.out = torch.nn.Linear(width, d_model)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """`g` (B, K, 3, T) gravity channels -> (B, K, d_model), added to the window summary."""
        q = max(1, g.shape[-1] // 4)
        feats = torch.cat((g.mean(-1), g[..., -q:].mean(-1) - g[..., :q].mean(-1)), dim=-1)
        return self.out(self.mlp(feats))


@register("two_stage")
class TwoStageModel(VelocityModel):
    """Encoder over windows, then context and embodiment across the chunk.

    Named "two stage" for the two levels it works at: inside a one-second window,
    and across the run of windows that make up a chunk.
    """

    def __init__(self, in_channels: int = 6, encoder: dict | None = None,
                 context: dict | None = None, embodiment: dict | None = None,
                 conditioner: dict | None = None, heads: dict | None = None,
                 dropout: float = 0.1, window: int = 200,
                 attitude: dict | None = None, **unused):
        super().__init__()
        self.in_channels = in_channels
        enc_cfg = dict(encoder or {})
        ctx_cfg = dict(context or {})
        emb_cfg = dict(embodiment or {})
        cond_cfg = dict(conditioner or {})
        head_cfg = dict(heads or {})
        # M21, AirIO's attitude branch. Off unless a config asks for it, and --
        # like the gravity head -- never declared in a reference, because every
        # key is inside the config hash. "late": the encoder reads the six raw
        # channels only and the gravity channels reach the model through the
        # branch alone (AirIO as published). "both": the encoder keeps all nine
        # and the branch is added on top.
        att_cfg = dict(attitude or {})
        self.attitude_mode = att_cfg.get("mode")
        if self.attitude_mode is not None:
            if self.attitude_mode not in ATTITUDE_MODES:
                raise KeyError(f"unknown model.attitude.mode {self.attitude_mode!r}; "
                               f"have {list(ATTITUDE_MODES)}")
            if in_channels < 9:
                raise ValueError("model.attitude needs the gravity channels (a nine-channel "
                                 f"input_repr), got {in_channels} channels")
        enc_channels = in_channels - 3 if self.attitude_mode == "late" else in_channels

        d_model = int(enc_cfg.pop("d_model", 256))
        self.d_model = d_model
        self.encoder = build_encoder(
            enc_cfg.pop("mode", "cnn_small"), in_channels=enc_channels,
            d_model=d_model, dropout=dropout, window=window, **enc_cfg)
        self.context = build_context(
            ctx_cfg.pop("mode", "identity"), d_model=d_model,
            dropout=dropout, **ctx_cfg)

        width = int(emb_cfg.pop("width", 32))
        self.embodiment = Embodiment(d_model=d_model, width=width, **emb_cfg)
        self.conditioner = build_conditioner(
            cond_cfg.pop("mode", "concat"), d_model=d_model, width=width, **cond_cfg)

        self.velocity_head = VelocityHead(d_model)
        # Optional so that M5b can build a copy without it and check that the
        # predictions are byte-identical.
        self.platform_head = (PlatformHead(width)
                              if head_cfg.get("platform", True) else None)
        # M15. Off unless a config asks for it, and the key is deliberately NOT
        # declared in any reference: every key is inside the config hash, so
        # declaring `gravity: false` would rehash every run in results.jsonl.
        # Built LAST, so that it draws its initial weights after everything
        # else has drawn theirs -- with the same seed, every other parameter is
        # then bit-identical to the model without it (tests/test_m15.py).
        self.gravity_head = (GravityHead(d_model)
                             if head_cfg.get("gravity", False) else None)
        # Built after everything else for the same reason: with the same seed,
        # every other parameter draws exactly what it draws without the branch.
        self.attitude = (AttitudeBranch(d_model, int(att_cfg.get("width", 64)))
                         if self.attitude_mode is not None else None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                cond: torch.Tensor) -> dict[str, torch.Tensor]:
        """`x` (B,K,C,T), `mask` (B,K) -> {"velocity": (B,K,3), "platform_logits": (B,4)}.

        `cond` is accepted to satisfy the shared model interface and is
        deliberately unused; see this module's docstring.
        """
        b, k, c, t = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")

        # Stage 1. Every window is encoded independently, so folding the chunk
        # and window axes together turns B*K separate calls into one.
        xe = x
        if self.attitude_mode == "late":
            # The encoder never sees the gravity channels; any extra scalars
            # after them still reach it.
            xe = torch.cat((x[:, :, :GRAVITY_CHANNELS.start], x[:, :, GRAVITY_CHANNELS.stop:]), dim=2)
        z = self.encoder(xe.reshape(b * k, xe.shape[2], t)).reshape(b, k, self.d_model)
        if self.attitude is not None:
            # AirIO joins the attitude to the IMU features before the recurrent
            # stage; everything downstream reads the joined summary.
            z = z + self.attitude(x[:, :, GRAVITY_CHANNELS])

        # Stage 3 reads the raw summaries, before context mixes them, so that the
        # M8 and M9 comparisons stay independent of each other.
        e = self.embodiment(z, mask)

        # Stage 2, then stage 4: share information along the chunk, then let the
        # embodiment vector adjust the result.
        context_out = self.context(z, mask)
        features = self.conditioner(context_out, e)

        out = {"velocity": self.velocity_head(features)}
        if self.platform_head is not None:
            # Produced last and never read back. The velocity above is already
            # final at this point -- that is the compliance rule, made structural.
            out["platform_logits"] = self.platform_head(e)
        if self.gravity_head is not None:
            # Same pattern. It reads the context output from *before* the
            # conditioner, per window, and nothing reads what it produces.
            out["gravity"] = self.gravity_head(context_out)
        out["embodiment"] = e
        return out
