"""Stage 1 of the model: turn one second of raw signal into one vector. models.md §3.

Each sample the model sees is a *chunk* -- a run of consecutive one-second
windows from a single recording. The encoder works on one window at a time: 200
frames of 6 or 9 channels go in, a single vector of `d_model` numbers comes out.
Think of it as writing a short summary of what the sensors did during that
second. Everything later in the network reads those summaries rather than the
raw signal.

**This is where almost all the compute lives.** Measured on this laptop
(models.md §3): a small convolutional encoder runs a full training step in
0.085 s on the GPU while a ResNet18 takes 0.210 s, and adding a whole
bidirectional context model on top of either costs only a few percent more. So
the encoder is the piece worth choosing carefully, and `cnn_small` is the
default because being five times faster means five times as many experiments in
a twenty-five-day budget.

**The four candidates, and what each one is for.** Milestone M6 races them
against each other; `bakeoff.py` in this folder runs the race.

| mode         | idea                                          | why it is in the race            |
| ------------ | --------------------------------------------- | -------------------------------- |
| `mlp`        | flatten the window, two dense layers          | the floor -- almost ignores time |
| `cnn_small`  | four strided convolutions, 200 frames -> 13   | the incumbent; fastest           |
| `resnet1d`   | a 1-D ResNet18, five times the parameters     | does capacity help?              |
| `tcn`        | dilated convolutions, full 200-frame rate     | does keeping the fine detail help? |

The three convolutional ones differ mainly in *what they throw away*.
`cnn_small` halves the length at every block, so by the end one position stands
for sixteen raw frames. `resnet1d` does the same but with far more channels to
describe each position. `tcn` never shortens anything: all 200 positions survive
to the end, and dilated filters reach across the whole second anyway. Since the
signal that separates a drone from a dog is high-frequency vibration, whether
that fine detail is worth keeping is a real question rather than a formality.

**Two rules here have measured reasons behind them.**

*Full rate, no decimation.* The released baseline throws away four frames in
every five with a plain `seg[::5]` and no anti-alias filter, which folds
everything above 20 Hz back onto the signal as false low frequencies. The 50-100
Hz accelerometer band that gets destroyed is the second most informative feature
for telling the four platforms apart. We read all 200 frames.

*LayerNorm, never BatchNorm.* BatchNorm normalises using statistics gathered
across the whole batch, and our batch is made of windows that are consecutive in
time and therefore very similar to each other. Those statistics describe the
particular stretch of recording that landed in the batch, not the dataset, so
they mislead. LayerNorm normalises each sample against itself across channels,
which has no such problem. This holds for every encoder below without exception,
and `tests/test_m6.py` checks it by walking the module tree of all four.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ENCODER_MODES = ("mlp", "cnn_small", "resnet1d", "tcn")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a `(N, C, L)` convolutional feature map.

    PyTorch's LayerNorm normalises over the *last* axes, but a 1-D convolution
    puts channels in the middle and time last. Rather than reason about that
    every time, this moves the axes, normalises, and moves them back. Two
    transposes on a tensor this size are free next to the convolution itself.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SmallCNNEncoder(nn.Module):
    """Four convolutional blocks that shorten 200 frames to 13, then average.

    Each block looks at a short stretch of time, mixes the channels, and then
    halves the length: 200 -> 100 -> 50 -> 25 -> 13. Stacking them means each
    output position summarises 91 raw frames, about half a second, even though
    every individual filter only spans seven -- the standard way a convolutional
    network sees far without a huge kernel.

    The final step averages over the 13 positions that remain. That makes the
    output length-independent and, more usefully, makes it a summary of the whole
    second rather than of its last moment.
    """

    def __init__(self, in_channels: int = 6, d_model: int = 256,
                 widths: tuple[int, ...] = (64, 128, 192, 256),
                 kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in widths:
            blocks += [
                nn.Conv1d(c_in, c_out, kernel_size, stride=2,
                          padding=kernel_size // 2),
                ChannelLayerNorm(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, C, T)` one window each -> `(N, d_model)`."""
        h = self.blocks(x)                 # (N, 256, 13) for a 200-frame window
        h = h.mean(dim=2)                  # average over what is left of time
        return self.dropout(self.project(h))


class MLPEncoder(nn.Module):
    """Flatten the whole window and push it through two dense layers.

    Deliberately the dumbest thing that could work. It exists so the encoder
    comparison at M6 has a floor to measure against -- without one, "the
    convolutional encoder scores X" is a number with nothing to be better than.
    It also ignores the ordering of the frames almost entirely, which makes it a
    useful check on how much the time structure is really worth.
    """

    def __init__(self, in_channels: int = 6, d_model: int = 256,
                 hidden: int = 512, window: int = 200, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------- resnet1d

class ResidualBlock1d(nn.Module):
    """Two convolutions plus a shortcut that skips over them.

    The shortcut is the whole idea of a residual network. Without it, the only
    way for information to reach the output is *through* every layer, and a deep
    stack of layers tends to blur or lose it on the way. With it, each block only
    has to learn what to *add* to what it was given, so a block that has nothing
    useful to contribute can settle on adding nothing and does no harm. That is
    what makes it safe to stack eight of them.

    When a block changes the number of channels or halves the length, the
    shortcut cannot be a plain copy -- the shapes would not match -- so a
    one-position convolution reshapes it to match. That is the standard
    "projection shortcut".
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1,
                 kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.norm1 = ChannelLayerNorm(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=pad, bias=False)
        self.norm2 = ChannelLayerNorm(c_out)
        self.act = nn.GELU()
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                ChannelLayerNorm(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class ResNet1dEncoder(nn.Module):
    """A 1-D ResNet18: a wide stem, then four stages of two residual blocks each.

    This is the same trunk shape the released baseline uses, which is exactly why
    it is in the race: it makes "our encoder against theirs" an honest
    comparison rather than a comparison of two unrelated things. The differences
    from the published version are deliberate and small -- LayerNorm instead of
    BatchNorm for the reason at the top of this file, and GELU instead of ReLU to
    match the rest of our network.

    The shape of one 200-frame window as it passes through:
    stem halves it to 100, the pooling step halves it again to 50, then the four
    stages leave it at 50, 25, 13, 7 while the channel count grows 64 -> 128 ->
    256 -> 512. Averaging the seven surviving positions gives one vector, which a
    linear layer maps to `d_model`.

    About five times the parameters of `cnn_small` and about two and a half times
    the time per step. Whether that buys anything is the M6 question.
    """

    def __init__(self, in_channels: int = 6, d_model: int = 256,
                 widths: tuple[int, ...] = (64, 128, 256, 512),
                 blocks_per_stage: int = 2, kernel_size: int = 3,
                 stem_kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        stem_width = widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_width, stem_kernel, stride=2,
                      padding=stem_kernel // 2, bias=False),
            ChannelLayerNorm(stem_width),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        c_in = stem_width
        for stage, c_out in enumerate(widths):
            for block in range(blocks_per_stage):
                # Only the first stage keeps the length; every later stage halves
                # it once, on its first block.
                stride = 2 if (stage > 0 and block == 0) else 1
                stages.append(ResidualBlock1d(c_in, c_out, stride, kernel_size))
                c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.project = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, C, T)` one window each -> `(N, d_model)`."""
        h = self.stages(self.stem(x))      # (N, 512, 7) for a 200-frame window
        h = h.mean(dim=2)
        return self.dropout(self.project(h))


# -------------------------------------------------------------------------- tcn

class DilatedBlock1d(nn.Module):
    """One dilated convolution with a shortcut, seeing a wide but sparse span.

    A *dilated* convolution spreads its filter out with gaps: with a dilation of
    eight, a three-position filter reads frames 0, 8 and 16 rather than 0, 1 and
    2. So it reaches sixteen frames wide while still costing three multiplies.
    Stacking blocks whose dilation doubles each time -- 1, 2, 4, 8, ... -- makes
    the reach grow exponentially with depth while the length of the signal is
    never reduced.

    The padding is put on *both* sides, so each output position sits in the
    middle of what it read. That is deliberate: we are allowed to look forward as
    well as back (models.md §4), and a lopsided filter would waste half its
    reach.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"dilated blocks need an odd kernel so the padding "
                             f"can be centred; got kernel_size={kernel_size}")
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = ChannelLayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.norm(self.conv(x))))


class TCNEncoder(nn.Module):
    """Dilated convolutions at the full 200 Hz rate -- nothing is ever shortened.

    The other two convolutional encoders buy their wide view by halving the
    length repeatedly, which means the fine detail of the signal is averaged away
    early. This one buys the same wide view by spreading its filters out instead,
    so all 200 positions are still there at the end. If the high-frequency
    vibration that distinguishes the platforms matters, this is the encoder that
    should show it.

    With seven blocks at dilations 1, 2, 4, ..., 64 and a three-position filter,
    the last block's view spans 255 frames -- comfortably more than the 200 in a
    window, so every output position has seen the entire second. `receptive_field`
    reports that number and `tests/test_m6.py` asserts it covers the window,
    because a TCN whose reach falls short of its input is a silently crippled
    model rather than a broken one.
    """

    def __init__(self, in_channels: int = 6, d_model: int = 256,
                 channels: int = 128, n_blocks: int = 7, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, 1, bias=False),
            ChannelLayerNorm(channels),
            nn.GELU(),
        )
        self.kernel_size = kernel_size
        self.dilations = tuple(2 ** i for i in range(n_blocks))
        self.blocks = nn.Sequential(*[
            DilatedBlock1d(channels, kernel_size, d, dropout)
            for d in self.dilations
        ])
        self.project = nn.Linear(channels, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def receptive_field(self) -> int:
        """How many raw frames one output position can see. Should exceed 200."""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, C, T)` one window each -> `(N, d_model)`."""
        h = self.blocks(self.stem(x))      # (N, 128, 200) -- still full length
        h = h.mean(dim=2)
        return self.dropout(self.project(h))


# ---------------------------------------------------------------------- factory

def build_encoder(mode: str = "cnn_small", in_channels: int = 6,
                  d_model: int = 256, dropout: float = 0.1,
                  window: int = 200, input_hz: int = 200, **extra) -> nn.Module:
    """Make the encoder named by `mode`. models.md §3 lists the candidates."""
    if input_hz != 200:
        raise NotImplementedError(
            "input_hz other than 200 is the decimation half of the M7 ablation and "
            "is not built. Note that decimating needs an anti-alias filter first -- "
            "the released baseline omits one, which is the defect models.md §3 "
            "documents, not a pattern to copy.")
    if mode == "cnn_small":
        return SmallCNNEncoder(in_channels, d_model, dropout=dropout, **extra)
    if mode == "mlp":
        return MLPEncoder(in_channels, d_model, window=window, dropout=dropout, **extra)
    if mode == "resnet1d":
        return ResNet1dEncoder(in_channels, d_model, dropout=dropout, **extra)
    if mode == "tcn":
        return TCNEncoder(in_channels, d_model, dropout=dropout, **extra)
    raise KeyError(f"unknown encoder mode {mode!r}; have {list(ENCODER_MODES)}")
