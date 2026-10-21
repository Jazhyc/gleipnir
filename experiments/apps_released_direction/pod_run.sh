#!/bin/bash
# Stage the hash-locked Lens overlay into the image runtime, then run the
# frozen direction study on the released 4B. Usage: pod_run.sh
set -euo pipefail
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/apps_released_direction
mkdir -p $L
if [ ! -d /tmp/gleipnir-vllm-lens-1.3.0/vllm_lens ]; then
  uv pip install --python $PY --no-deps --require-hashes --link-mode copy \
    --target /tmp/gleipnir-vllm-lens-1.3.0 -r experiments/b200_vllm_lens/requirements.txt
fi
(cd /tmp/gleipnir-vllm-lens-1.3.0 && find . -type f -not -path '*__pycache__*' | sort | xargs sha256sum) > $L/lens_overlay.sha256
$PY -m experiments.apps_released_direction.run > $L/run.log 2>&1
echo "DIRECTION_DONE $(date -Is)"
