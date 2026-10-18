#!/usr/bin/env bash
# Canary, then the fast-GDN and last-query candidates, one server at a time.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
MODEL=/tmp/gleipnir-merged/qwen38-27b-s0-01
RECEIPT=results/qwen38_27b_serving/last_query_bf16.json
echo "canary_start $(date -Is)"
$PY -m experiments.qwen38_27b_serving.last_query_bf16 --model $MODEL --output $RECEIPT
echo "canary_done $(date -Is)"
$PY -m experiments.qwen38_27b_serving.bench run fastgdn01 \
  --worker gleipnir.serving.bf16_stack.Bf16FastGdnWorker
BINDING=$(python3 -c "import json,hashlib,sys; p='$RECEIPT'; print(json.dumps({'last_query_attention': {'validation': p, 'validation_sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()}}))")
$PY -m experiments.qwen38_27b_serving.bench run lastquery01 \
  --worker gleipnir.serving.bf16_stack.Bf16LastQueryWorker --additional "$BINDING"
$PY -m experiments.qwen38_27b_serving.bench run lqfast01 \
  --worker gleipnir.serving.bf16_stack.Bf16LastQueryFastGdnWorker --additional "$BINDING"
echo QUEUE_DONE
