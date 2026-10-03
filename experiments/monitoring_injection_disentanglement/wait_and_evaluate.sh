#!/usr/bin/env bash
# Serialize evaluation behind the existing training process; this is not monitoring.
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
training_pid=$(cat results/monitoring_injection_disentanglement/train.pid)
while kill -0 "$training_pid" 2>/dev/null; do
    sleep 30
done
test -f results/monitoring_injection_disentanglement/4b/conservative/complete.json
exec .venv/bin/python -m experiments.monitoring_injection_disentanglement.launch_evaluate
