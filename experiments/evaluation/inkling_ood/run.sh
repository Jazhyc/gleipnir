#!/usr/bin/env bash
# Score on the Pod with the vLLM nightly runtime.
# Usage: run.sh inkling_effort_none|inkling_effort_high ; CANARY_ONLY=1 stops after the canary.
set -euo pipefail
cd /workspace/gleipnir
PY=/root/inkling-vllm/bin/python
export HF_HOME=/root/hf-inkling HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir
# No network volume on this Pod: compiler caches live on the container disk.
C=/root/cache/inkling_ood
export PATH=$(dirname $PY):/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
L=logs/runpod/inkling_ood; mkdir -p $L
EXTRA=${CANARY_ONLY:+--stop-after-canary}
SUFFIX=${CANARY_ONLY:+_canary}
case $1 in
  inkling_effort_none|inkling_effort_high) NAME=$1 ;;
  *) echo "unknown run $1"; exit 2 ;;
esac
echo "start $NAME$SUFFIX $(date -Is)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.used --format=csv,noheader > $L/gpu_$NAME$SUFFIX.txt
$PY -m experiments.evaluation.inkling_ood.score --config experiments/evaluation/inkling_ood/$NAME.json \
  --output results/inkling_ood/$NAME $EXTRA > $L/$NAME$SUFFIX.log 2>&1
echo "done $NAME$SUFFIX $(date -Is)"
