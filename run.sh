#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.11}"
if [[ ! -d .venv ]]; then "$PYTHON" -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p runtime/logs
python -u src/pipeline.py "$@" 2>&1 | tee -a runtime/logs/launcher.log
