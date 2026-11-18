#!/usr/bin/env bash
# Trains stage 1 and stage 2 on the GPU, exports them and packs runs/upload_*.zip.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then echo "run bash gpu_training/setup_linux.sh first"; exit 1; fi
exec .venv/bin/python gpu_training/train_all.py "$@"
