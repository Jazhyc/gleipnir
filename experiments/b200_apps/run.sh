#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
export PYTHONPATH=src:.
py=/tmp/gleipnir-vllm031-runtime/bin/python
run_name=${1:-apps02}
"$py" -m experiments.b200_vllm031.runtime -m experiments.b200_apps.run --stage prepare --name "$run_name"
"$py" -m experiments.b200_vllm031.runtime -m experiments.b200_attention_precision.startup --name apps_default02
"$py" -m experiments.b200_vllm031.runtime -m experiments.b200_apps.run --stage score --name "$run_name"
"$py" -m experiments.b200_vllm031.runtime -m experiments.b200_apps.run --stage analyze --name "$run_name"
