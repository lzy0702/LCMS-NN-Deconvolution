#!/usr/bin/env bash
# Trains the runs (by default the window and the comb view) on the GPU, exports them and packs
# runs/upload_*.zip (see gpu_training/train_all.py; --runs chooses the runs).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then echo "run bash gpu_training/setup_linux.sh first"; exit 1; fi
exec .venv/bin/python gpu_training/train_all.py "$@"
