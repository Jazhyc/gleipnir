#!/usr/bin/env bash
# Gleipnir 27B OOD through the recommended optimized stack (graphs01 flags:
# QuantFullWorker, FP8/FP4 geometry, MXFP8 attention, graphs to 2,048 tokens).
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
FULL=$(merge "$(bind last_query_attention $OUT/last_query_mxfp8.json)" "$(bind mxfp8_attention $OUT/mxfp8_canary01.json)" \
  "$(bind fp4_gemm_tuning $OUT/fp4_gemm_tune01.json)" "$(bind fp4_producers $OUT/fp4_producers01.json)")
W=gleipnir.serving.quant_stack.QuantFullWorker
SIZES=$(python3 -c "import json; s=[1,2,4]+list(range(8,257,8))+[320,384,448,512,640,768,896,1024,1280,1536,1792,2048]; print(json.dumps({'cudagraph_capture_sizes': s}))")
echo "start $(date -Is)"
$PY -m experiments.optimized_gap.ood27b ${1:-opt27b01} --quantization $Q --worker $W \
  --additional "$FULL" --compilation "$SIZES" --prompt ${2:-teacher}
echo "QUEUE_DONE $(date -Is)"
