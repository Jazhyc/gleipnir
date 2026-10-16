#!/usr/bin/env bash
# Run the 320-row ten-update packing benchmark: control, then final-layer trim.
# Uses the NC2 training runtime and kernel builds restored on the EU Pod.
set -euo pipefail
cd /workspace/gleipnir
export HF_HOME=/workspace/gleipnir/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir PYTHONUNBUFFERED=1 WANDB_MODE=disabled
for condition in control trim; do
  echo "screen_start $condition $(date -Is)"
  /tmp/gleipnir-training-runtime/venv/bin/python experiments/fp4_stability/run.py \
    --config experiments/b200_training_trim/screen_$condition.yaml
  echo "screen_done $condition $(date -Is)"
done
echo SCREENS_DONE
