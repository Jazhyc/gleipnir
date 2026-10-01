#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export FLA_DISABLE_BACKEND_DISPATCH=1
ln -sfn "$PWD/.cache/kernels/fla" /tmp/gleipnir-qwen35-fla
ln -sfn "$PWD/.cache/kernels/causal_conv1d" /tmp/gleipnir-qwen35-causal-conv1d
ln -sfn "$PWD/.cache/kernels/triton" /tmp/gleipnir-triton-3.7.1
exec .venv/bin/python experiments/fp4_stability/run.py \
  --config "${1:-experiments/fp4_stability/config.yaml}"
