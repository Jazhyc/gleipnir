#!/bin/bash
# Per-action replay on a Pod running the released serving image (no volume),
# plus the image 1.2 release check. For each model: serve in the default mode
# (no prefix caching), check identity scores against the 1.0 B200 rows
# (REFERENCE) and smoke-test, replay; then restart with --prefix-caching,
# smoke-test and replay again. One server at a time. The first server starts
# with PUBLIC_KEY set, to check that the 1.2 entrypoint starts SSH beside it.
# Usage: REFERENCE=results/release_serving/image10/fresh pod_run.sh NAME [4b] [27b]
set -euo pipefail
NAME=$1; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(4b 27b)
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
L=logs/runpod/prefix_caching_replay/$NAME
mkdir -p $L
SERVER=
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null; true' EXIT
for M in "${MODELS[@]}"; do
  for MODE in uncached cached; do
    FLAGS=""; [ $MODE = cached ] && FLAGS="--prefix-caching"
    LOG=$L/serve_${M}_$MODE.log
    KEY=""; [ $M = "${MODELS[0]}" ] && [ $MODE = uncached ] && KEY="${SSH_PUBLIC_KEY:-}"
    PUBLIC_KEY="$KEY" /opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh $M --port 8000 $FLAGS > $LOG 2>&1 &
    SERVER=$!
    until grep -q gleipnir_serve_ready $LOG; do
      kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M $MODE"; tail -40 $LOG; exit 1; }
      sleep 5
    done
    echo "SERVER_READY $M $MODE $(date -Is)"
    if [ -n "$KEY" ]; then
      [ -e /var/log/gleipnir-ssh-start.log ] && echo "SSH_START_RAN" || { echo "NO_SSH_START"; exit 1; }
    fi
    SMOKE=""; [ $MODE = cached ] && SMOKE="--prefix-caching"
    $PY -m experiments.release_serving.image.smoke --port 8000 $SMOKE | tail -1
    if [ $MODE = uncached ]; then
      $PY -m experiments.release_serving.image.identity --output $L/${M}_identity.jsonl --compare $REFERENCE/${M}_default.jsonl | tail -1
      $PY -m experiments.serving.prefix_caching_replay.probe --model $M --name $NAME | tail -2
    fi
    $PY -m experiments.serving.prefix_caching_replay.replay --model $M --mode $MODE --name $NAME > $L/replay_${M}_$MODE.log 2>&1
    echo "REPLAYED $M $MODE $(date -Is)"
    kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  done
done
echo "ALL_DONE $(date -Is)"
