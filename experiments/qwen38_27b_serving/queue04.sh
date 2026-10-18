#!/usr/bin/env bash
# Latency floor: the full stack with piecewise CUDA graphs extended from 256 to
# 2,048 tokens (graphs01) against default capture (full02), 3 c1 passes each.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
OUT=results/qwen38_27b_serving
Q=gleipnir_geometry_fp8_fp4
bind() {
  python3 -c "import json,hashlib,sys; k,p=sys.argv[1:]; print(json.dumps({k: {'validation': p, 'validation_sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()}}))" "$1" "$2"
}
merge() { python3 -c "import json,sys; d={}; [d.update(json.loads(a)) for a in sys.argv[1:]]; print(json.dumps(d))" "$@"; }
step() { echo "$1 $(date -Is)"; }
FULL=$(merge "$(bind last_query_attention $OUT/last_query_mxfp8.json)" "$(bind mxfp8_attention $OUT/mxfp8_canary01.json)" \
  "$(bind fp4_gemm_tuning $OUT/fp4_gemm_tune01.json)" "$(bind fp4_producers $OUT/fp4_producers01.json)")
W=gleipnir.serving.quant_stack.QuantFullWorker
# vLLM 0.31's default sizes up to 256, then coarser steps (padding <= 25%).
SIZES=$(python3 -c "import json; s=[1,2,4]+list(range(8,257,8))+[320,384,448,512,640,768,896,1024,1280,1536,1792,2048]; print(json.dumps({'cudagraph_capture_sizes': s}))")
step full02_start
$PY -m experiments.qwen38_27b_serving.bench run full02 --quantization $Q --worker $W \
  --additional "$FULL" --repeats 1 --c1-repeats 3
step graphs01_start
$PY -m experiments.qwen38_27b_serving.bench run graphs01 --quantization $Q --worker $W \
  --additional "$FULL" --compilation "$SIZES" --repeats 3 --c1-repeats 3
echo QUEUE_DONE
