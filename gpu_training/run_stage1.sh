#!/usr/bin/env bash
# Stage 1: LadderNet (full size), singly charged ions kept in the spectra but ignored by the loss.
# Usage: bash gpu_training/run_stage1.sh [--resume] [workers]
#   --resume   continue a paused or interrupted run
#   workers    processes generating spectra (default: number of cores minus 2)
set -euo pipefail
cd "$(dirname "$0")/.."
RESUME=""
if [ "${1:-}" = "--resume" ]; then RESUME="--resume"; shift; fi
WORKERS="${1:-$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) - 2 ))}"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged ignore --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers "$WORKERS" --out runs/ladder_full_ignore.pt $RESUME
