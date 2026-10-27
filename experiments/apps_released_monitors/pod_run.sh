#!/bin/bash
# Score both released models on a Pod running the released serving image (no
# volume): for each model start the image entrypoint, score the frozen APPS
# prompts once, stop the server. No ID passes (user decision, 2026-10-21).
# Usage: pod_run.sh NAME
set -euo pipefail
NAME=$1
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/apps_released_monitors
mkdir -p $L
for M in 4b 27b; do
  /opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 > $L/serve_${NAME}_$M.log 2>&1 &
  SERVER=$!
  until grep -q gleipnir_serve_ready $L/serve_${NAME}_$M.log; do
    kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; exit 1; }
    sleep 5
  done
  echo "SERVER_READY $M $(date -Is)"
  $PY -m experiments.apps_released_monitors.run score $M --name $NAME > $L/score_${NAME}_$M.log 2>&1
  echo "SCORED $M $(date -Is)"
  kill $SERVER; wait $SERVER 2>/dev/null || true
  pkill -f "release_serving.serve" || true
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
    pkill -f "vllm" || true
    sleep 5
  done
done
echo "ALL_DONE $(date -Is)"
