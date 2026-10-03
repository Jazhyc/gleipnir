#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m experiments.monitoring_injection_removal.prepare
exec .venv/bin/python -m experiments.monitoring_injection_removal.train
