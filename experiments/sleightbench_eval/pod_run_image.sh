#!/bin/bash
# Score one model on a Pod running the released serving image (no volume):
# start the image entrypoint at its default 262K, score the frozen variant
# once, stop. No ID passes (user decision, 2026-10-21); the gate receipt is
# the sb01 one for this unchanged stack.
# Usage: pod_run_image.sh MODEL NAME VARIANT GATE
set -euo pipefail
M=$1; NAME=$2; VARIANT=$3; GATE=$4
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/sleightbench_eval
mkdir -p $L
/opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 > $L/serve_${NAME}_$M.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
until grep -q gleipnir_serve_ready $L/serve_${NAME}_$M.log; do
  kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; exit 1; }
  sleep 5
done
echo "SERVER_READY $M $(date -Is)"
$PY -m experiments.sleightbench_eval.run --variant $VARIANT score $M --name $NAME \
  --validation $GATE > $L/score_${NAME}_$M.log 2>&1
echo "SCORED $M $(date -Is)"
