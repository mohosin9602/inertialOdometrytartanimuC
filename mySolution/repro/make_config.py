#!/usr/bin/env python3
"""Build a runnable config for the released baseline.

Works around a bug in the shipped starter script: tartanimu_submission.py reads
cfg["model_param"] at line 168, but that key is only merged into the config by
configer.build_model() at line 171 -- three lines later. The HF unified.yaml
does not contain it, so the script dies with KeyError: 'model_param'.

Fix without touching any tracked file: pre-merge the organizers' own model yaml
into their own unified.yaml, exactly as build_model() would, and hand the result
to the script's documented --config flag.

Also rewrites model.model_yaml to an absolute path, so build_model() no longer
requires you to be cd'd into the TartanIMU repo root.
"""
import logging
import os

import yaml
from huggingface_hub import hf_hub_download

from paths import OUT, add_import_paths

add_import_paths()
# importing tartan_imu installs a DEBUG-level root logger, which floods stdout
# with huggingface_hub HTTP traces; silence it before it takes effect.
from tartan_imu.config import configer  # noqa: E402 # TartanIMU folder gets added as add_import_paths() is executed. Before that, pylance cant see it in paths. Thus, the warning.

logging.disable(logging.INFO)

HF_REPO = "Tartan-IMU/TartanIMU"


def main():
    os.makedirs(OUT, exist_ok=True)

    unified = hf_hub_download(HF_REPO, "config/unified.yaml")
    model_yaml = hf_hub_download(HF_REPO, "config/resnet_lstm_multihead.yaml")

    cfg = configer.load_config(unified)
    special = yaml.load(open(model_yaml), Loader=yaml.Loader)
    configer.update_recursive(cfg, special)      # identical to build_model()'s merge

    # absolute, so the config no longer depends on the current working directory
    cfg["model"]["model_yaml"] = model_yaml
    cfg["train"]["use_multi_gpu"] = False

    out = os.path.join(OUT, "unified_merged.yaml")
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)

    win = int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"])
    print(f"wrote {out}")
    print(f"  model_param merged: {'model_param' in cfg}")
    print(f"  window = {win} frames  (expect 200)")
    assert win == 200, "unexpected window size"


if __name__ == "__main__":
    main()
