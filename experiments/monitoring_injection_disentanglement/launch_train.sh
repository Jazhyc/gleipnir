#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m experiments.monitoring_injection_disentanglement.prepare
exec .venv/bin/python -m experiments.monitoring_injection_disentanglement.train
