#!/bin/bash
# Stock BF16 reference arms on a Pod running the released serving image:
# fetch the released 27B merged checkpoint, then score the base and released
# 27B in one offline engine each. Usage: pod_offline.sh NAME
set -euo pipefail
NAME=$1
cd /workspace/gleipnir
export PYTHONPATH=src:.
export PATH=/tmp/gleipnir-vllm031-runtime/bin:$PATH
export HF_HOME=/workspace/.hf
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/apps_released_monitors
mkdir -p $L
$PY -m experiments.release_serving.fetch 27b > $L/fetch_${NAME}_27b.log 2>&1
for ARM in 27b_base_bf16 27b_release_bf16; do
  $PY -m experiments.apps_released_monitors.offline $ARM --name $NAME > $L/offline_${NAME}_$ARM.log 2>&1
  echo "SCORED $ARM $(date -Is)"
done
echo "ALL_DONE $(date -Is)"
