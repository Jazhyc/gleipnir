#!/usr/bin/env bash
# Phase 2 kernel receipts; they read only the model config, so they can run
# on the pinned base while the merged checkpoint is being rebuilt.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:.
MODEL=${1:-.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
OUT=results/qwen38_27b_serving
echo "mxfp8_canary_start $(date -Is)"
$PY -m experiments.qwen38_27b_serving.mxfp8_canary --model $MODEL --output $OUT/mxfp8_canary01.json
echo "last_query_mxfp8_start $(date -Is)"
$PY -m experiments.qwen38_27b_serving.last_query_mxfp8 --model $MODEL --output $OUT/last_query_mxfp8.json
echo "gemm_tune_start $(date -Is)"
$PY -m experiments.qwen38_27b_serving.fp4_gemm_tune --model $MODEL --output $OUT/fp4_gemm_tune01.json
echo RECEIPTS_DONE
