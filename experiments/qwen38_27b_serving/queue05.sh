#!/usr/bin/env bash
# Native FP4 SwiGLU output: canary with band selection, then the full stack
# with native output and extended graphs (native01) and its ID drift.
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
step swiglu_canary_start
[ -f $OUT/fp4_swiglu_native02.json ] || $PY -m experiments.qwen38_27b_serving.fp4_swiglu_native_canary --model $MODEL --output $OUT/fp4_swiglu_native02.json
python3 -c "import json; d=json.load(open('$OUT/fp4_swiglu_native02.json')); print('selection', d['state'], d['passed'], d.get('selection')); assert d['passed'] and d['selection']['minimum_rows']"
NATIVE=$(merge "$(bind last_query_attention $OUT/last_query_mxfp8.json)" "$(bind mxfp8_attention $OUT/mxfp8_canary01.json)" \
  "$(bind fp4_gemm_tuning $OUT/fp4_gemm_tune01.json)" "$(bind fp4_producers $OUT/fp4_producers01.json)" \
  "$(bind fp4_swiglu_native $OUT/fp4_swiglu_native02.json)")
W=gleipnir.serving.quant_stack.QuantNativeWorker
SIZES=$(python3 -c "import json; s=[1,2,4]+list(range(8,257,8))+[320,384,448,512,640,768,896,1024,1280,1536,1792,2048]; print(json.dumps({'cudagraph_capture_sizes': s}))")
step native01_start
$PY -m experiments.qwen38_27b_serving.bench run native01 --quantization $Q --worker $W \
  --additional "$NATIVE" --compilation "$SIZES" --repeats 3 --c1-repeats 3
step nativeid01_start
$PY -m experiments.qwen38_27b_serving.id_drift nativeid01 --quantization $Q --worker $W \
  --additional "$NATIVE" --compilation "$SIZES"
echo QUEUE_DONE
