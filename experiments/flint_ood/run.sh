#!/usr/bin/env bash
# Score on the Pod with the vLLM 0.34.0 runtime. Usage: run.sh flint
# CANARY_ONLY=1 stops after the canary.
set -euo pipefail
cd /workspace/gleipnir
PY=/root/flint-vllm034/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir
# No network volume on this Pod: compiler caches live on the container disk.
C=/root/cache/flint_ood
export PATH=$(dirname $PY):/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
export FLASHINFER_WORKSPACE_BASE=$C/flashinfer VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
L=logs/runpod/flint_ood; mkdir -p $L
EXTRA=${CANARY_ONLY:+--stop-after-canary}
SUFFIX=${CANARY_ONLY:+_canary}
case $1 in
  flint)
    export HF_HOME=/root/hf-flint
    MODULE=experiments.flint_ood.score CONFIG=flint.json NAME=flint ;;
  *) echo "unknown run $1"; exit 2 ;;
esac
echo "start $NAME$SUFFIX $(date -Is)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.used --format=csv,noheader > $L/gpu_$NAME$SUFFIX.txt
$PY -m $MODULE --config experiments/flint_ood/$CONFIG \
  --output results/flint_ood/$NAME $EXTRA > $L/$NAME$SUFFIX.log 2>&1
echo "done $NAME$SUFFIX $(date -Is)"
