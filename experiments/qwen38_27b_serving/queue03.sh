#!/usr/bin/env bash
# Phase 2 continued: fused-producer receipt, the full stack (MXFP8 attention,
# tuned tiles, fused producers), then its ID drift against BF16.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
MODEL=/tmp/gleipnir-merged/qwen38-27b-s0-01
OUT=results/qwen38_27b_serving
Q=gleipnir_geometry_fp8_fp4
bind() {
  python3 -c "import json,hashlib,sys; k,p=sys.argv[1:]; print(json.dumps({k: {'validation': p, 'validation_sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()}}))" "$1" "$2"
}
merge() { python3 -c "import json,sys; d={}; [d.update(json.loads(a)) for a in sys.argv[1:]]; print(json.dumps(d))" "$@"; }
step() { echo "$1 $(date -Is)"; }

step producers_canary_start
[ -f $OUT/fp4_producers01.json ] || $PY -m experiments.qwen38_27b_serving.fp4_producers_canary --model $MODEL --output $OUT/fp4_producers01.json
FULL=$(merge "$(bind last_query_attention $OUT/last_query_mxfp8.json)" "$(bind mxfp8_attention $OUT/mxfp8_canary01.json)" \
  "$(bind fp4_gemm_tuning $OUT/fp4_gemm_tune01.json)" "$(bind fp4_producers $OUT/fp4_producers01.json)")
step full01_start
$PY -m experiments.qwen38_27b_serving.bench run full01 --quantization $Q \
  --worker gleipnir.serving.quant_stack.QuantFullWorker --additional "$FULL"
step fullid01_start
$PY -m experiments.qwen38_27b_serving.id_drift fullid01 --quantization $Q \
  --worker gleipnir.serving.quant_stack.QuantFullWorker --additional "$FULL"
echo QUEUE_DONE
