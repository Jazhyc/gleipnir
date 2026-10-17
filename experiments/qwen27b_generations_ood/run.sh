#!/usr/bin/env bash
# Score the three frozen 27B configs in order on the Pod (vLLM 0.31, BF16).
set -euo pipefail
cd /workspace/gleipnir
export HF_HOME=/root/hf-27b HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir PYTHONUNBUFFERED=1
PY=/tmp/gleipnir-vllm031-runtime/bin/python
# Plain vLLM 0.31 (no serving overlays): ninja/CUDA for JIT kernels and
# persistent, experiment-local compiler caches on the volume.
C=/workspace/gleipnir/.cache/qwen27b_generations_ood
export PATH=$(dirname $PY):/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
L=logs/runpod/qwen27b_generations_ood
EXTRA=${CANARY_ONLY:+--stop-after-canary}
for name in ${@:-qwen35_27b qwen36_27b qwen38_27b}; do
  echo "start $name $(date -Is)"
  $PY -m experiments.tool_trajectory_monitoring.benchmark_qwen_ood \
    --config experiments/qwen27b_generations_ood/$name.json \
    --output results/qwen27b_generations_ood/$name${CANARY_ONLY:+_canary} $EXTRA \
    > $L/$name${CANARY_ONLY:+_canary}.log 2>&1
  echo "done $name $(date -Is)"
done
echo RUNS_DONE
