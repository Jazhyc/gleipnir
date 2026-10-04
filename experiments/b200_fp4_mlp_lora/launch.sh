#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .cache-runtime.env
exec .venv/bin/python -m experiments.b200_fp4_mlp_lora.run "$@"
