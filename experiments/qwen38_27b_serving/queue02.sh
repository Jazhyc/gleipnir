#!/usr/bin/env bash
# Phase 2 on a new Pod: same-GPU controls, MXFP8 and tuning receipts, then
# the MXFP8 and MXFP8 + tuned-GEMM candidates, one GPU job at a time.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
MODEL=/tmp/gleipnir-merged/qwen38-27b-s0-01
OUT=results/qwen38_27b_serving
Q=gleipnir_geometry_fp8_fp4
bind() {  # bind KEY RECEIPT -> {"KEY": {"validation": ..., "validation_sha256": ...}}
  python3 -c "import json,hashlib,sys; k,p=sys.argv[1:]; print(json.dumps({k: {'validation': p, 'validation_sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()}}))" "$1" "$2"
}
merge() { python3 -c "import json,sys; d={}; [d.update(json.loads(a)) for a in sys.argv[1:]]; print(json.dumps(d))" "$@"; }
step() { echo "$1 $(date -Is)"; }
once() {  # once RECEIPT CMD...: skip receipts already produced (e.g. pre-merge)
  if [ -f "$1" ]; then echo "reuse $1"; else shift; "$@"; fi
}

LQ_BF16=$(bind last_query_attention $OUT/last_query_bf16.json)
step control02_start
$PY -m experiments.qwen38_27b_serving.bench run control02
step quant02_start
$PY -m experiments.qwen38_27b_serving.bench run quant02 --quantization $Q \
  --worker gleipnir.serving.quant_stack.QuantLastQueryFastGdnWorker --additional "$LQ_BF16"

step mxfp8_canary_start
once $OUT/mxfp8_canary01.json $PY -m experiments.qwen38_27b_serving.mxfp8_canary --model $MODEL --output $OUT/mxfp8_canary01.json
step last_query_mxfp8_start
once $OUT/last_query_mxfp8.json $PY -m experiments.qwen38_27b_serving.last_query_mxfp8 --model $MODEL --output $OUT/last_query_mxfp8.json
step gemm_tune_start
once $OUT/fp4_gemm_tune01.json $PY -m experiments.qwen38_27b_serving.fp4_gemm_tune --model $MODEL --output $OUT/fp4_gemm_tune01.json

MX=$(merge "$(bind last_query_attention $OUT/last_query_mxfp8.json)" "$(bind mxfp8_attention $OUT/mxfp8_canary01.json)")
step mxfp801_start
$PY -m experiments.qwen38_27b_serving.bench run mxfp801 --quantization $Q \
  --worker gleipnir.serving.quant_stack.QuantMxfp8LastQueryFastGdnWorker --additional "$MX"
TUNED=$(merge "$MX" "$(bind fp4_gemm_tuning $OUT/fp4_gemm_tune01.json)")
step tuned01_start
$PY -m experiments.qwen38_27b_serving.bench run tuned01 --quantization $Q \
  --worker gleipnir.serving.quant_stack.QuantTunedMxfp8LastQueryFastGdnWorker --additional "$TUNED"
echo QUEUE_DONE
