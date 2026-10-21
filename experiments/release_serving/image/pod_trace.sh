#!/bin/bash
# Serve one target with the open() audit trace on, in the default (262K) mode
# and with --prefix-caching, smoke-test each launch, then stop it and remove
# the hook. ID agreement passes are not repeated (user decision, 2026-10-21):
# the 262K stack passed them in experiments/sleightbench_eval run sb01.
# Usage: pod_trace.sh {4b|27b}
set -euo pipefail
M=$1
cd /workspace/gleipnir
L=logs/runpod/release_serving
PY=/tmp/gleipnir-vllm031-runtime/bin/python
SITE=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p $L /tmp/gleipnir-trace-mod
cp experiments/release_serving/image/open_trace.py /tmp/gleipnir-trace-mod/gleipnir_open_trace.py
echo "import sys; sys.path.insert(0, '/tmp/gleipnir-trace-mod'); import gleipnir_open_trace" > $SITE/zz_gleipnir_open_trace.pth
SERVER=
trap 'rm -f $SITE/zz_gleipnir_open_trace.pth; [ -n "$SERVER" ] && kill $SERVER 2>/dev/null; true' EXIT
for MODE in default prefix; do
  FLAGS=""; SMOKE=""
  if [ $MODE = prefix ]; then FLAGS="--prefix-caching"; SMOKE="--prefix-caching"; fi
  LOG=$L/trace_${M}_$MODE.log
  mkdir -p /tmp/gleipnir-trace/${M}_$MODE
  GLEIPNIR_OPEN_TRACE=/tmp/gleipnir-trace/${M}_$MODE PYTHONPATH=src:. $PY -m experiments.release_serving.serve $M $FLAGS --host 127.0.0.1 --port 8000 > $LOG 2>&1 &
  SERVER=$!
  started=$(date +%s)
  until grep -q gleipnir_serve_ready $LOG; do
    kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M $MODE"; tail -40 $LOG; exit 1; }
    sleep 5
  done
  echo "ready $M $MODE after $(( $(date +%s) - started )) s"
  PYTHONPATH=src:. $PY -m experiments.release_serving.image.smoke --port 8000 $SMOKE | tail -1
  kill $SERVER; wait $SERVER || true; SERVER=
done
echo "TRACE_DONE $M"
