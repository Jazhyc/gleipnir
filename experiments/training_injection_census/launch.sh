#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p logs/runpod/training_injection_census
exec .venv/bin/python -m experiments.training_injection_census.run
