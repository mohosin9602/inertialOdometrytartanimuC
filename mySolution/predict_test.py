"""Write a test-split submission from a finished run's weights.

This is the inference path of the submitted model, on its own: it reads the run's
config.json and weight file, builds the test dataset from the raw-NPZ caches, runs one
forward pass per chunk and writes window_id,vx,vy,vz. No training, no labels.

Before the first use, build the caches from the raw test NPZ files:
    python -m src.prep.windows --split test        # window index cache
    python -m src.prep.orientation --split test    # complementary-filter gravity channels

Usage (from mySolution/):
    python predict_test.py runs/20260919-m23-phase-trainval-seed42-b8e7089e
    python predict_test.py <run-dir> --weights model.pt --device cpu --out submission.csv
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from src.pipeline import build_dataset, build_model, predict_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory holding config.json and the weights")
    ap.add_argument("--weights", default="model.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    a = ap.parse_args()

    cfg = copy.deepcopy(json.load(open(a.run / "config.json"))["config"])
    cfg["run"]["device"] = a.device
    ds = build_dataset(cfg, "test")
    model = build_model(cfg, in_channels=ds.n_channels)
    sd = torch.load(a.run / a.weights, map_location="cpu")
    model.load_state_dict(sd.get("state_dict", sd))
    table, stats = predict_dataset(model, ds, cfg, verbose=False)
    table.sort_values("window_id").to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(table)} rows from {stats['n_chunks']} chunks")


if __name__ == "__main__":
    main()
