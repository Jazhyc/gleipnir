#!/usr/bin/env bash
# Serialize scoring; this process does not provide agent monitoring.
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
training_pid=$(cat results/monitoring_injection_removal/train.pid)
while kill -0 "$training_pid" 2>/dev/null; do
    if [[ -r /proc/$training_pid/stat ]] && [[ $(awk '{print $3}' /proc/$training_pid/stat) == Z ]]; then
        break
    fi
    sleep 30
done
test -f results/monitoring_injection_removal/4b/filtered/complete.json
exec .venv/bin/python -m experiments.monitoring_injection_removal.launch_evaluate
