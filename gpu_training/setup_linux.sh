#!/usr/bin/env bash
# One-time setup of the training environment on Linux (see gpu_training/setup.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=""
for v in 3.12 3.13 3.11 3.14; do
    if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
PY="${PY:-python3}"
exec "$PY" gpu_training/setup.py "$@"
