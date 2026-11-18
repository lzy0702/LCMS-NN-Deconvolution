#!/usr/bin/env bash
# Stage 1 only (gpu_training/train_all.sh runs both stages and packs the results).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python gpu_training/train_all.py --only 1 "$@"
