#!/bin/bash
# Score the EvilGenie runs with assistant messages on a Pod running image 1.2
# (no volume), with the released-stack scorer of evilgenie_released_stack.
# Usage: pod_run.sh [4b] [27b]
set -euo pipefail
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(4b 27b)
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
CFG=experiments/evilgenie_assistant_messages/config.json
L=logs/runpod/evilgenie_assistant_messages
mkdir -p $L
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; true' EXIT
for M in "${MODELS[@]}"; do
  LOG=$L/serve_$M.log
  /opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 > $LOG 2>&1 &
  SERVER=$!
  until grep -q gleipnir_serve_ready $LOG; do
    kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; tail -40 $LOG; exit 1; }
    sleep 5
  done
  echo "SERVER_READY $M $(date -Is)"
  $PY -m experiments.release_serving.image.smoke --port 8000 | tail -1
  $PY -m experiments.evilgenie_released_stack.pod workload --model $M --config $CFG 2>&1 | tee $L/workload_$M.log | tail -1
  $PY -m experiments.evilgenie_released_stack.pod score --model $M --config $CFG 2>&1 | tee $L/score_$M.log | tail -1
  echo "SCORED $M $(date -Is)"
  kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=
done
echo "ALL_DONE $(date -Is)"
