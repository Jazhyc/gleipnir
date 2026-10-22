#!/bin/bash
# One model on the B200: serve at 262K, two ID passes, repeat-relative gate,
# then the single SleightBench pass. Stops at the first failure.
set -euo pipefail
M=$1
NAME=$2
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/sleightbench_eval
V=results/sleightbench_eval/$NAME/validation_$M
mkdir -p $L $V
$PY -m experiments.release_serving.fetch $M > $L/fetch_$M.log 2>&1
$PY -m experiments.release_serving.serve $M --host 127.0.0.1 --port 8000 > $L/serve_$M.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
until grep -q gleipnir_serve_ready $L/serve_$M.log; do
  kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; exit 1; }
  sleep 5
done
echo "SERVER_READY $M $(date -Is)"
for p in a b; do
  $PY -m experiments.release_serving.validate $M --port 8000 --splits id \
    --output $V/$p.json --rows-output $V/$p > $L/validate_${M}_$p.log 2>&1 || true
  tail -1 $L/validate_${M}_$p.log
done
$PY -m experiments.release_serving.gate $M --pass-a $V/a.json --pass-b $V/b.json \
  --output $V/gate.json | tail -1
$PY -m experiments.sleightbench_eval.run score $M --name $NAME --validation $V/gate.json \
  > $L/score_$M.log 2>&1
echo "SCORED $M $(date -Is)"
