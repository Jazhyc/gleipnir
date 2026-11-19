#!/bin/bash
# Gleipnir optimized B200 serving image: fetch a release checkpoint, then serve.
# Usage: <image> {4b|27b} [--port 8000] [--host 0.0.0.0] [--prefix-caching ...]
#        <image> <command> [args]   run another command instead, e.g. /start.sh or bash
# Since 1.2: with PUBLIC_KEY set (Runpod's convention) and the base image's
# /start.sh present, SSH starts in the background before the server.
set -euo pipefail
case "${1:-}" in
  "" | 4b | 27b) ;;
  *) exec "$@" ;;
esac
if [ -n "${PUBLIC_KEY:-}" ] && [ -x /start.sh ]; then
  /start.sh > /var/log/gleipnir-ssh-start.log 2>&1 &
fi
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
