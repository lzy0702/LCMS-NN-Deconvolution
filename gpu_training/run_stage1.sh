#!/usr/bin/env bash
# Stage 1: LadderNet (full size), singly charged ions kept in the spectra but ignored by the loss.
# Usage: bash gpu_training/run_stage1.sh [--resume] [--workers N] [other train_charge_net.py options]
#   --resume       continue a paused or interrupted run
#   --workers N    processes generating spectra (default: cores minus 2, at most 8)
set -euo pipefail
cd "$(dirname "$0")/.."
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$(( CORES > 3 ? CORES - 2 : 1 ))
if [ "$WORKERS" -gt 8 ]; then WORKERS=8; fi
# options given on the command line come last and win (argparse keeps the last value)
python scripts/train_charge_net.py --device cuda --arch ladder --model-size full \
    --singly-charged ignore --epochs 30 --train-len 4000 --val-len 128 --batch-size 8 \
    --workers "$WORKERS" --out runs/ladder_full_ignore.pt "$@"
