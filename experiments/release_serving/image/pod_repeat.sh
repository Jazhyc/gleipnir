#!/bin/bash
# Two back-to-back validation passes on one server, saving per-row scores, to
# measure the stack's own run-to-run variation on this host.
set -euo pipefail
M=$1
cd /workspace/gleipnir
L=logs/runpod/release_serving
PY=/tmp/gleipnir-vllm031-runtime/bin/python
PYTHONPATH=src:. $PY -m experiments.release_serving.serve $M --host 127.0.0.1 --port 8000 > $L/serve_${M}_repeat.log 2>&1 &
SERVER=$!
until grep -q gleipnir_serve_ready $L/serve_${M}_repeat.log; do
  kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; exit 1; }
  sleep 5
done
for pass in a b; do
  PYTHONPATH=src:. $PY -m experiments.release_serving.validate $M --port 8000 \
    --output results/release_serving/$M/repeat_$pass.json \
    --rows-output results/release_serving/$M/repeat_$pass > $L/validate_${M}_repeat_$pass.log 2>&1 || true
  tail -1 $L/validate_${M}_repeat_$pass.log
done
kill $SERVER; wait $SERVER || true
echo "REPEAT_DONE $M"
