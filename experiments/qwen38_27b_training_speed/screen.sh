#!/usr/bin/env bash
# Run the 320-row ten-update packed benchmark on Qwen3.8-27B, one config at a time.
# Uses the NC2 training runtime and kernel builds restored on the EU Pod; the
# 27B weights live on the container disk under /root/hf-27b.
set -euo pipefail
cd /workspace/gleipnir
export HF_HOME=/root/hf-27b HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir PYTHONUNBUFFERED=1 WANDB_MODE=disabled
for condition in "$@"; do
  echo "screen_start $condition $(date -Is)"
  /tmp/gleipnir-training-runtime/venv/bin/python experiments/fp4_stability/run.py \
    --config experiments/qwen38_27b_training_speed/screen_$condition.yaml
  echo "screen_done $condition $(date -Is)"
done
echo SCREENS_DONE
