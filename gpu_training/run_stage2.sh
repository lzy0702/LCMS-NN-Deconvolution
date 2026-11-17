#!/usr/bin/env bash
# Stage 2: continue stage 1 with singly charged ions trained on and dense chemical background, so
# the network learns to recognise z = 1 background instead of reading it as multiply charged.
# Usage: bash gpu_training/run_stage2.sh [--resume] [--workers N] [other train_charge_net.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f runs/ladder_full_ignore.pt ]; then
    echo "runs/ladder_full_ignore.pt not found: run stage 1 (gpu_training/run_stage1.sh) first" >&2
    exit 1
fi
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$(( CORES > 3 ? CORES - 2 : 1 ))
if [ "$WORKERS" -gt 8 ]; then WORKERS=8; fi
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged keep --background-ions 200 --init runs/ladder_full_ignore.pt \
    --epochs 12 --lr 5e-4 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers "$WORKERS" --out runs/ladder_full_stage2.pt "$@"
