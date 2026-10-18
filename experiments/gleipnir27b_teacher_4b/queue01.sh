#!/usr/bin/env bash
# Gleipnir 27B under the teacher prompt on the canonical CoT-removed ID rows
# (teacher reference), then both 4B arms in turn.
set -euo pipefail
cd /workspace/gleipnir
export PYTHONPATH=src:. HF_HUB_OFFLINE=1
echo "teacher_reference_start $(date -Is)"
/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime \
  -m experiments.gleipnir27b_vocab_distillation.extract --model gleipnir \
  --jobs teacher:id_cot_removed --name extract01
echo "arms_start $(date -Is)"
/tmp/gleipnir-training-runtime/venv/bin/python -m experiments.gleipnir27b_teacher_4b.run
echo QUEUE_DONE
