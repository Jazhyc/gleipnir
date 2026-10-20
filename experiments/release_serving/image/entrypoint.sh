#!/bin/bash
# Gleipnir optimized B200 serving image: fetch a release checkpoint, then serve.
# Usage: <image> {4b|27b} [--port 8000] [--host 0.0.0.0]
set -euo pipefail
MODEL="${1:-${GLEIPNIR_MODEL:-4b}}"
shift || true
for path in /tmp/gleipnir-vllm031-runtime /tmp/gleipnir-serving-runtime /tmp/gleipnir-gigatoken-0.10.0; do
  [ -e "$path" ] || { echo "missing $path: do not mount a volume over /tmp" >&2; exit 1; }
done
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
"$PY" -m experiments.release_serving.fetch "$MODEL"
exec "$PY" -m experiments.release_serving.serve "$MODEL" "$@"
