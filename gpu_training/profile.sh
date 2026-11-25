#!/usr/bin/env bash
# Measures this computer for the developers and packs runs/profile_upload_*.zip (see
# gpu_training/profile_machine.py); the setup runs first where it no longer holds.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=""
for v in 3.12 3.13 3.11 3.14; do
    if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
PY="${PY:-python3}"
"$PY" gpu_training/setup.py --if-needed
export LCMSDECONV_SETUP_RC=$?
if [ ! -x .venv/bin/python ]; then echo "the setup did not make .venv: see gpu_training/setup.log"; exit 1; fi
exec .venv/bin/python gpu_training/profile_machine.py "$@"
