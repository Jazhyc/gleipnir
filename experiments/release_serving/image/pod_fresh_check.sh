#!/bin/bash
# Fresh-Pod check of a pushed image (no volume): for each model, run the
# image's own entrypoint cold (download, verify, serve), smoke-test the
# default 262K server, then restart it with --prefix-caching and smoke-test
# the cache. No ID passes (user decision, 2026-10-21).
# Usage: pod_fresh_check.sh [4b] [27b]
set -euo pipefail
cd /workspace/gleipnir
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=/root/fresh_check
mkdir -p $L
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; true' EXIT
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(4b 27b)
for M in "${MODELS[@]}"; do
  for MODE in default prefix; do
    FLAGS=""; [ $MODE = prefix ] && FLAGS="--prefix-caching"
    LOG=$L/${M}_$MODE.log
    started=$(date +%s)
    /opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 $FLAGS > $LOG 2>&1 &
    SERVER=$!
    until grep -q gleipnir_serve_ready $LOG; do
      kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M $MODE"; tail -40 $LOG; exit 1; }
      sleep 5
    done
    echo "ready $M $MODE after $(( $(date +%s) - started )) s"
    PYTHONPATH=src:. $PY -m experiments.release_serving.image.smoke --port 8000 $FLAGS | tail -1
    kill $SERVER; wait $SERVER || true; SERVER=
  done
done
echo FRESH_CHECK_DONE
