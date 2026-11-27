#!/usr/bin/env bash
# Fetch the three pinned BF16 checkpoints to the Pod's container disk.
set -euo pipefail
export HF_HOME=/root/hf-27b HF_HUB_DISABLE_TELEMETRY=1
PY=/tmp/gleipnir-serving-runtime/venv/bin/python
for spec in "Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654" \
            "Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9" \
            "Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"; do
  set -- $spec
  ( $PY -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2', max_workers=16))" \
      > /workspace/gleipnir/logs/runpod/qwen27b_generations_ood/download_${1#Qwen/}.log 2>&1 \
    && echo "done $1 $(date -Is)" || echo "FAILED $1 $(date -Is)" ) &
done
wait
du -sh $HF_HOME/hub/* 2>/dev/null
echo DOWNLOADS_DONE
