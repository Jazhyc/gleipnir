#!/bin/bash
# Fresh-Pod check of a pushed image (no volume). For each model:
# - run the image's own entrypoint cold (download, verify, serve) with the
#   recorded graphs and the Transformers tokenizer, and score the synthetic
#   identity prompts one at a time;
# - restart in the default mode (larger graphs, Gigatoken), smoke-test it,
#   require identical identity scores and a native tokenizer receipt;
# - restart with --prefix-caching and smoke-test the cache.
# No ID passes (user decision, 2026-10-21).
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
  for MODE in recorded default prefix; do
    FLAGS=""
    [ $MODE = recorded ] && FLAGS="--recorded-graphs --transformers-tokenizer"
    [ $MODE = prefix ] && FLAGS="--prefix-caching"
    LOG=$L/${M}_$MODE.log
    rm -f /tmp/gleipnir-release/$M/tokenizer.json
    started=$(date +%s)
    /opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 $FLAGS > $LOG 2>&1 &
    SERVER=$!
    until grep -q gleipnir_serve_ready $LOG; do
      kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M $MODE"; tail -40 $LOG; exit 1; }
      sleep 5
    done
    echo "ready $M $MODE after $(( $(date +%s) - started )) s"
    export PYTHONPATH=src:.
    if [ $MODE = recorded ]; then
      $PY -m experiments.release_serving.image.identity --output $L/${M}_recorded.jsonl | tail -1
    else
      SMOKE=""; [ $MODE = prefix ] && SMOKE="--prefix-caching"
      $PY -m experiments.release_serving.image.smoke --port 8000 $SMOKE | tail -1
    fi
    if [ $MODE = default ]; then
      $PY -m experiments.release_serving.image.identity --output $L/${M}_default.jsonl --compare $L/${M}_recorded.jsonl | tail -1
      grep -q gigatoken_native /tmp/gleipnir-release/$M/tokenizer.json || { echo "NO_NATIVE_TOKENIZER $M"; exit 1; }
      cp /tmp/gleipnir-release/$M/tokenizer.json $L/${M}_tokenizer.json
    fi
    kill $SERVER; wait $SERVER || true; SERVER=
  done
done
echo FRESH_CHECK_DONE
