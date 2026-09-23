"""Tier 0 -- a learned gyroscope denoiser (M19, lever 4). Brossard et al., RA-L 2020.

    mySolution/.venv/bin/python -m src.prep.gyro_denoise           # train on train, report val

**What it is.** Brossard, Bonnabel and Barrau showed that a small dilated
convolutional network, reading a few seconds of IMU around each instant, can
learn a per-frame correction to a cheap gyroscope good enough that integrating
the corrected rate open-loop beats visual-inertial attitude. A better gyro means
the gravity estimators, which carry the accelerometer across seconds with the
gyro, stay accurate through manoeuvres. Here: corrected = C @ gyro + net(imu),
C a learned 3x3 calibration (starts at the identity), the network's output
starts near zero.

**What it learns from.** Ground-truth body rate, from consecutive `quat`
samples, expressed in the gyro's OWN frame (the per-recording axis map fitted
the way `orientation.gyro_frame_from_truth` fits it). So this is pure denoising
and calibration: it does not try to fix the drone gyro-frame defect (finding
21), which is lever 3's question, and the two stay separable in the report.
Ground truth is a TRAINING target only, which the organizers confirmed is
allowed; at inference the denoiser reads the six raw axes and nothing else.

**Its legal standing is weaker than the physics estimators'.** It is a second
set of learned weights at inference, in front of the velocity model. Nothing in
it is per-platform, so "one shared set of weights" in the sense of the
2026-08-30 ruling (no per-platform experts) plausibly holds -- but ask the host
before a submission depends on it. As a report ablation it is fine.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..paths import CACHE_V, PLATFORMS, traj_npz
from ..utils import env_record, write_json
from .cache import load_cache

DT = 1.0 / 200.0
HORIZONS = (16, 32, 64)          # frames the loss averages rates over: 80-320 ms
CROP = 1024                      # frames per training crop (5 s)
CAP = 40_000                     # frames kept per training recording
SEED = 20260918


def denoiser_path(root: Path | None = None) -> Path:
    return (root or CACHE_V) / "gyro_denoiser.pt"


class GyroDenoiser(nn.Module):
    """Dilated 1-D convolutions over (acc, gyro) -> a per-frame gyro correction."""

    def __init__(self, width: int = 64, kernel: int = 7,
                 dilations: tuple = (1, 4, 16, 64), scale: float = 0.05):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 6
        for d in dilations:
            layers += [nn.Conv1d(c_in, width, kernel, dilation=d, padding=d * (kernel // 2)),
                       nn.GELU()]
            c_in = width
        layers.append(nn.Conv1d(width, 3, 1))
        self.net = nn.Sequential(*layers)
        self.C = nn.Parameter(torch.eye(3))
        self.scale = float(scale)
        self.register_buffer("mean", torch.zeros(6))
        self.register_buffer("std", torch.ones(6))
        #: frames of context either side; inference pads chunks by this much
        self.reach = sum(d * (kernel // 2) for d in dilations)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """`imu` (B, T, 6) in m/s^2 and rad/s -> corrected gyro (B, T, 3), rad/s."""
        x = ((imu - self.mean) / self.std).transpose(1, 2)
        return imu[..., 3:6] @ self.C.T + self.net(x).transpose(1, 2) * self.scale


# ---------------------------------------------------------------------- the data

def _body_rate(quat: np.ndarray) -> np.ndarray:
    """Ground-truth body rate per frame, central difference; ends repeated."""
    from .orientation import _conj, _logmap, _qmul
    q = np.asarray(quat, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w = _logmap(_qmul(_conj(q[:-2]), q[2:])) / (2 * DT)
    return np.concatenate((w[:1], w, w[-1:]))


def _own_frame(gyro: np.ndarray, w_gt: np.ndarray) -> np.ndarray:
    """The truth rate carried into the gyro's own axes (nearest signed permutation)."""
    from .orientation import _nearest_perm
    k = 10
    c1 = np.cumsum(np.vstack((np.zeros((1, 3)), gyro)), axis=0)
    c2 = np.cumsum(np.vstack((np.zeros((1, 3)), w_gt)), axis=0)
    a, b = (c1[k:] - c1[:-k]) / k, (c2[k:] - c2[:-k]) / k
    u, _, vt = np.linalg.svd(b.T @ a)
    Og = _nearest_perm(u @ vt)                     # gyro -> truth
    return w_gt @ Og                               # Og^T w, row-wise


def load_items(split: str, cap: int = CAP) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(platform, imu (N, 6) float32, target rate (N, 3) float32) per recording."""
    cache = load_cache(split)
    out = []
    for t in range(cache.n_traj):
        r = cache.trajectories.iloc[t]
        n = min(int(r.n_frames_kept), cap)
        lo = int(r.frame_offset)
        imu = np.asarray(cache.imu[lo:lo + n], np.float32)
        raw = np.load(traj_npz(split, str(r.platform), str(r.traj_id)))
        w = _own_frame(imu[:, 3:6].astype(np.float64), _body_rate(raw["quat"][:n]))
        out.append((str(r.platform), imu, w.astype(np.float32)))
    return out


def _horizon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brossard's idea on rates: compare means over several horizons, robustly."""
    loss = 0.0
    for h in HORIZONS:
        cp = torch.cumsum(pred, 1)
        ct = torch.cumsum(target, 1)
        mp = (cp[:, h:] - cp[:, :-h]) / h
        mt = (ct[:, h:] - ct[:, :-h]) / h
        loss = loss + nn.functional.smooth_l1_loss(mp, mt, beta=0.01)
    return loss / len(HORIZONS)


# ---------------------------------------------------------------------- training

def train(steps: int = 3000, batch: int = 16, lr: float = 1e-3, device: str = "cpu",
          root: Path | None = None, verbose: bool = True) -> dict:
    """Fit on train, platform-balanced crops; save weights next to the caches."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    items = [it for it in load_items("train") if len(it[1]) > CROP + 1]
    by_p = {p: [i for i, it in enumerate(items) if it[0] == p] for p in PLATFORMS}
    model = GyroDenoiser()
    allf = np.concatenate([it[1][::50] for it in items])
    model.mean.copy_(torch.from_numpy(allf.mean(0)))
    model.std.copy_(torch.from_numpy(np.maximum(allf.std(0), 1e-3)))
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.perf_counter()
    for step in range(steps):
        xs, ys = [], []
        for _ in range(batch):
            p = PLATFORMS[int(rng.integers(len(PLATFORMS)))]
            _, imu, w = items[int(rng.choice(by_p[p]))]
            s = int(rng.integers(0, len(imu) - CROP))
            xs.append(imu[s:s + CROP]); ys.append(w[s:s + CROP])
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        loss = _horizon_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"  step {step:5d}/{steps}  loss {loss.item():.5f}  "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    model.cpu().eval()
    path = denoiser_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "steps": steps, "seed": SEED}, path)
    meta = {"steps": steps, "batch": batch, "lr": lr, "crop": CROP, "cap": CAP,
            "horizons": HORIZONS, "seed": SEED, "device": device,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "seconds": round(time.perf_counter() - t0, 1), "built_by": env_record()}
    write_json(path.with_suffix(".json"), meta)
    return meta


def load_denoiser(root: Path | None = None) -> GyroDenoiser:
    path = denoiser_path(root)
    if not path.exists():
        raise FileNotFoundError(f"no gyro denoiser at {path}. Train it with:\n"
                                f"    mySolution/.venv/bin/python -m src.prep.gyro_denoise")
    model = GyroDenoiser()
    model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return model.eval()


@torch.no_grad()
def denoise(imu: np.ndarray, model: GyroDenoiser, chunk: int = 50_000) -> np.ndarray:
    """`(N, 6)` IMU -> the same with the gyro columns replaced by the denoised rate.

    Long recordings run in pieces with `reach` frames of overlap each side, so
    the answer equals one pass over the whole recording.
    """
    x = np.asarray(imu, np.float32)
    n, r = len(x), model.reach
    out = np.empty((n, 3), np.float32)
    for s in range(0, n, chunk):
        a, b = max(0, s - r), min(n, s + chunk + r)
        y = model(torch.from_numpy(x[a:b])[None])[0].numpy()
        out[s:min(n, s + chunk)] = y[s - a:s - a + min(chunk, n - s)]
    res = np.asarray(imu, np.float64).copy()
    res[:, 3:6] = out
    return res


# ---------------------------------------------------------------------- report

def rate_report(model: GyroDenoiser, split: str = "val") -> dict:
    """Rate error over 32-frame means, raw vs denoised, per platform (rad/s)."""
    rep = {}
    for p, imu, w in load_items(split):
        den = denoise(imu, model)[:, 3:6]
        for name, g in (("raw", imu[:, 3:6].astype(np.float64)), ("denoised", den)):
            cg = np.cumsum(np.vstack((np.zeros((1, 3)), g)), 0)
            cw = np.cumsum(np.vstack((np.zeros((1, 3)), w)), 0)
            e = np.linalg.norm((cg[32:] - cg[:-32]) - (cw[32:] - cw[:-32]), axis=1) / 32
            rep.setdefault(p, {}).setdefault(name, []).append(e)
    return {p: {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in d.items()}
            for p, d in rep.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    meta = train(steps=a.steps, device=a.device)
    print(f"trained: {meta['n_parameters']:,} parameters, {meta['seconds']} s")
    for p, d in rate_report(load_denoiser()).items():
        print(f"  val {p:<6} rate error (32-frame means): raw {d['raw']:.4f}  "
              f"denoised {d['denoised']:.4f} rad/s")


if __name__ == "__main__":
    main()
