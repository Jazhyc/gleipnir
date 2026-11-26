#!/bin/bash
# Score the stripped OOD workloads on a Pod running the released image 1.2
# (no volume). One server at a time, default serving mode, smoke test first.
# Usage: pod_run.sh NAME [4b] [27b]
set -euo pipefail
NAME=$1; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(4b 27b)
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/tool_output_ablation/$NAME
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
  $PY -m experiments.tool_output_ablation.score --model $M --name $NAME 2>&1 | tee $L/score_$M.log | tail -1
  echo "SCORED $M $(date -Is)"
  kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=
done
echo "ALL_DONE $(date -Is)"
