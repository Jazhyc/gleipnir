#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
mkdir -p logs/runpod/runpod_gleipnir4b_id
nohup .venv/bin/python -u -m experiments.runpod_gleipnir4b_id.run run \
    > logs/runpod/runpod_gleipnir4b_id/launcher.log 2>&1 < /dev/null &
echo "$!" > results/runpod_gleipnir4b_id/launcher.pid
cat results/runpod_gleipnir4b_id/launcher.pid
