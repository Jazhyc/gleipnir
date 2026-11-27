#!/usr/bin/env bash
set -euo pipefail
d=$(cd "$(dirname "$0")" && pwd); while [ ! -f "$d/pyproject.toml" ]; do [ "$d" = / ] && { echo "no checkout above $0" >&2; exit 1; }; d=$(dirname "$d"); done
cd "$d"
source .cache-runtime.env
exec .venv/bin/python -m experiments.systems.b200_fp4_mlp_lora.run "$@"
