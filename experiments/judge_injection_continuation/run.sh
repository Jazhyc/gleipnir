#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m experiments.judge_injection_continuation.prepare
.venv/bin/python -m experiments.judge_injection_continuation.train
.venv/bin/python -m experiments.judge_injection_continuation.launch_evaluate
