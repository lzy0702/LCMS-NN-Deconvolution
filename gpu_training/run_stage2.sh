#!/usr/bin/env bash
# Stage 2 only, continuing from stage 1 (gpu_training/train_all.sh runs both stages).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python gpu_training/train_all.py --only 2 "$@"
