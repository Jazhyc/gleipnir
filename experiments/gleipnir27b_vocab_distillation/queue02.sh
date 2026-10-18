#!/usr/bin/env bash
# After queue01: Gleipnir 27B on the frozen OOD suite under the full teacher
# prompt (user request), then the step 1 diagnostic.
set -euo pipefail
cd /workspace/gleipnir
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
echo "gleipnir_ood_start $(date -Is)"
$PY -m experiments.gleipnir27b_vocab_distillation.extract --model gleipnir \
  --jobs teacher:ood --name extract01
echo "diagnostic_start $(date -Is)"
.venv/bin/python -m experiments.gleipnir27b_vocab_distillation.diagnose --name extract01
echo QUEUE_DONE
