#!/usr/bin/env bash
# Long context, 1K-1M at c1: the 27B MXFP8 1M envelope receipt, then stock
# BF16 and the optimized stack, native RoPE (to 256K) then YaRN (to 1M).
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
MODEL=/tmp/gleipnir-merged/qwen38-27b-s0-01
OUT=results/qwen38_27b_serving
step() { echo "$1 $(date -Is)"; }
step envelope_start
[ -f $OUT/mxfp8_envelope01.json ] || $PY -m experiments.qwen38_27b_serving.mxfp8_envelope --model $MODEL --output $OUT/mxfp8_envelope01.json
for run in "optimized native" "optimized yarn" "bf16 native" "bf16 yarn"; do
  set -- $run
  step "${1}_${2}_start"
  $PY -m experiments.qwen38_27b_serving.long_context long01 $1 $2
done
echo QUEUE_DONE
