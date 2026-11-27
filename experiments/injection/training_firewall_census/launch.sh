#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/gleipnir-matplotlib
exec /tmp/gleipnir-vllm031-runtime/bin/python \
  -m gleipnir.serving.stack.environment \
  -m experiments.injection.training_firewall_census.run
