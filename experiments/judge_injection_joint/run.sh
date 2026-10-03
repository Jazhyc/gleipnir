#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m experiments.judge_injection_joint.prepare
.venv/bin/python -m experiments.judge_injection_joint.train
.venv/bin/python -m experiments.judge_injection_joint.launch_evaluate
