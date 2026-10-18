#!/usr/bin/env bash
# Step 1 extraction: Gleipnir 27B, then unadapted Qwen3.8-27B, one engine each.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
for model in gleipnir base; do
  echo "${model}_start $(date -Is)"
  $PY -m experiments.gleipnir27b_vocab_distillation.extract --model $model \
    --jobs student:id teacher:id teacher:train --name extract01
done
echo QUEUE_DONE
