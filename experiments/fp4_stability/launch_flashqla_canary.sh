#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="$PWD/.cache/kernels/flashqla-da06429:/tmp/gleipnir-triton-3.7.1:/tmp/gleipnir-qwen35-fla:$PYTHONPATH"
export TILELANG_CACHE_DIR="$PWD/.cache/training/flashqla/tilelang"
export OMP_NUM_THREADS=4
exec .venv/bin/python experiments/fp4_stability/flashqla_canary.py "$@"
