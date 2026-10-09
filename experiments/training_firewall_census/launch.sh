#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/gleipnir-matplotlib
exec /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.training_firewall_census.run
