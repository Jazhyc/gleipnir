#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export FLA_DISABLE_BACKEND_DISPATCH=1
for mapping in 'fla gleipnir-qwen35-fla' 'causal_conv1d gleipnir-qwen35-causal-conv1d' 'triton gleipnir-triton-3.7.1'; do
    read -r persistent_name temporary_name <<< "$mapping"
    target="/workspace/gleipnir/.cache/kernels/$persistent_name"
    link="/tmp/$temporary_name"
    if [[ -e "$link" || -L "$link" ]]; then
        [[ "$(readlink -f "$link")" == "$target" ]]
    else
        ln -s "$target" "$link"
    fi
done
mkdir -p logs/runpod/b200_training_throughput
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_training_throughput/config.yaml \
    > logs/runpod/b200_training_throughput/preparation.log 2>&1
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_training_throughput/resolved_config.json \
    > logs/runpod/b200_training_throughput/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_training_throughput/launcher.pid
cat results/b200_training_throughput/launcher.pid
