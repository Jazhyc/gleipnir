#!/usr/bin/env bash
# Score the three frozen 27B configs in order on the Pod (vLLM 0.31, BF16).
set -euo pipefail
cd /workspace/gleipnir
export HF_HOME=/root/hf-27b HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir PYTHONUNBUFFERED=1
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/qwen27b_generations_ood
for name in ${@:-qwen35_27b qwen36_27b qwen38_27b}; do
  echo "start $name $(date -Is)"
  $PY -m experiments.tool_trajectory_monitoring.benchmark_qwen_ood \
    --config experiments/qwen27b_generations_ood/$name.json \
    --output results/qwen27b_generations_ood/$name > $L/$name.log 2>&1
  echo "done $name $(date -Is)"
done
echo RUNS_DONE
