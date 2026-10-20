#!/bin/bash
# Serve one target with the open() audit trace on, validate it on the build
# Pod (this also warms the compiler caches), then stop it and remove the hook.
# Usage: pod_trace.sh {4b|27b}
set -euo pipefail
M=$1
cd /workspace/gleipnir
L=logs/runpod/release_serving
PY=/tmp/gleipnir-vllm031-runtime/bin/python
SITE=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p /tmp/gleipnir-trace/$M /tmp/gleipnir-trace-mod results/release_serving/$M
cp experiments/release_serving/image/open_trace.py /tmp/gleipnir-trace-mod/gleipnir_open_trace.py
echo "import sys; sys.path.insert(0, '/tmp/gleipnir-trace-mod'); import gleipnir_open_trace" > $SITE/zz_gleipnir_open_trace.pth
trap 'rm -f $SITE/zz_gleipnir_open_trace.pth' EXIT
GLEIPNIR_OPEN_TRACE=/tmp/gleipnir-trace/$M PYTHONPATH=src:. $PY -m experiments.release_serving.serve $M --host 127.0.0.1 --port 8000 > $L/serve_$M.log 2>&1 &
SERVER=$!
started=$(date +%s)
until grep -q gleipnir_serve_ready $L/serve_$M.log; do
  kill -0 $SERVER 2>/dev/null || { echo "SERVER_EXITED $M"; tail -40 $L/serve_$M.log; exit 1; }
  sleep 5
done
echo "ready $M after $(( $(date +%s) - started )) s"
PYTHONPATH=src:. $PY -m experiments.release_serving.validate $M --port 8000 \
  --output results/release_serving/$M/build_validation.json > $L/validate_$M.log 2>&1 || true
tail -3 $L/validate_$M.log
kill $SERVER; wait $SERVER || true
echo "TRACE_DONE $M"
