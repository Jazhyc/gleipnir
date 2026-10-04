#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m experiments.monitor_injection_augmentation.audit
.venv/bin/python -m experiments.monitor_injection_augmentation.train
.venv/bin/python -m experiments.monitor_injection_augmentation.launch_evaluate
