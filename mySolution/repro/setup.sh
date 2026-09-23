#!/usr/bin/env bash 
# A typical shell script begins with a shebang (#.!), which instructs the operating system on which interpreter to use to read the file.
#? One-time environment setup for reproducing the TartanIMU baseline.
# Creates mySolution/.venv and installs the vendored TartanIMU library into it.
# Run from anywhere:  bash mySolution/repro/setup.sh
set -euo pipefail

REPRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYSOLUTION="$(dirname "$REPRO")"
ROOT="$(dirname "$MYSOLUTION")"
TARTAN="$ROOT/TartanIMU"
VENV="$MYSOLUTION/.venv"

[ -d "$TARTAN" ] || { echo "ERROR: $TARTAN not found (clone superxslam/TartanIMU there)"; exit 1; }

# --system-site-packages reuses an already-installed torch instead of pulling ~2 GB again.
python3 -m venv --system-site-packages "$VENV"

"$VENV/bin/pip" install -q --upgrade pip
# --no-deps: the heavy deps (torch, numpy, scipy...) come from system site-packages.
"$VENV/bin/pip" install -q -e "$TARTAN" --no-deps
# rich: a real runtime dep of tartan_imu.
# wandb: listed only as the optional "logging" extra, but tartan_imu.config.configer
#        imports train.py which imports wandb unconditionally -- so inference needs it.
# huggingface_hub: downloads the released baseline weights.
"$VENV/bin/pip" install -q rich wandb huggingface_hub pandas

# pip install -e leaves a build artifact inside the vendored repo; drop it so
# TartanIMU/ stays byte-for-byte pristine.
rm -rf "$TARTAN/tartan_imu.egg-info"

"$VENV/bin/python" -c "import tartan_imu, rich, wandb, torch, pandas; print('env OK, torch', torch.__version__)"
echo
echo "Done. Virtualenv: $VENV"
