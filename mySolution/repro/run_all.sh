#!/usr/bin/env bash
# Reproduce the released TartanIMU baseline and self-score it on the val split.
# Run from anywhere:  bash mySolution/repro/run_all.sh
# Prerequisite:       bash mySolution/repro/setup.sh
set -euo pipefail

REPRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYSOLUTION="$(dirname "$REPRO")"
ROOT="$(dirname "$MYSOLUTION")"
TARTAN="$ROOT/TartanIMU"
DATA="$MYSOLUTION/data/raw"
OUT="$REPRO/out"
PY="$MYSOLUTION/.venv/bin/python"
SUBMIT="$TARTAN/starter/tartanimu_submission.py"

DEVICE="${DEVICE:-cpu}"     # DEVICE=cuda:0 to use a GPU

[ -x "$PY" ] || { echo "ERROR: no venv. Run: bash $REPRO/setup.sh"; exit 1; }
[ -d "$DATA/index" ] || { echo "ERROR: competition data not found at $DATA"; exit 1; }

mkdir -p "$OUT"

# PYTHONPATH: tartan_imu.config.configer does `import train`, a repo-root script that
# pyproject.toml excludes from the installed package. Without the repo root on the path
# the import fails, because Python puts the *script's* dir (starter/) on sys.path.
export PYTHONPATH="$TARTAN"
export WANDB_MODE=disabled
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

echo "==> [1/6] merge the model config (works around the model_param KeyError)"
"$PY" "$REPRO/make_config.py"

echo
echo "==> [2/6] flatten val/ (starter expects <root>/<traj_id>.npz, val is nested)"
"$PY" "$REPRO/flatten_val.py"

echo
echo "==> [3/6] all-zero submission (pipeline check, 30644 rows)"
"$PY" "$TARTAN/starter/baseline_submission.py" \
    --sample_submission "$DATA/sample_submission.csv" \
    --out "$OUT/submission_zero.csv"

echo
echo "==> [4/6] baseline on val, heads routed by the CSV's platform column (~1 min CPU)"
"$PY" "$SUBMIT" \
    --test_root "$OUT/val_flat" \
    --windows   "$DATA/index/val_windows.csv" \
    --config    "$OUT/unified_merged.yaml" \
    --out       "$OUT/submission_val.csv" \
    --device    "$DEVICE" 2>&1 | grep -E "^loaded|^wrote"

echo
echo "==> [5/6] baseline on val with --head human forced (what the test split gets)"
"$PY" "$SUBMIT" \
    --test_root "$OUT/val_flat" \
    --windows   "$DATA/index/val_windows.csv" \
    --config    "$OUT/unified_merged.yaml" \
    --head      human \
    --out       "$OUT/submission_val_headhuman.csv" \
    --device    "$DEVICE" 2>&1 | grep -E "^loaded|^wrote"

echo
echo "==> [6/6] baseline on the anonymized test split -> uploadable CSV (~1 min CPU)"
"$PY" "$SUBMIT" \
    --test_root "$DATA/test" \
    --windows   "$DATA/index/test_windows.csv" \
    --config    "$OUT/unified_merged.yaml" \
    --head      human \
    --out       "$OUT/submission_tartanimu.csv" \
    --device    "$DEVICE" 2>&1 | grep -E "^loaded|^wrote"

echo
echo "================ SELF-SCORING ON VAL ================"
"$PY" "$REPRO/score_val.py"

rm -rf "$TARTAN/tartan_imu.egg-info"
echo
echo "Artifacts in $OUT"
echo "Upload to Kaggle: $OUT/submission_tartanimu.csv"
