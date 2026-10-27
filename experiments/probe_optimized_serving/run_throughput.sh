#!/usr/bin/env bash
# Throughput configurations on one B200, each on a fresh compiled server, all
# on the 6,395 frozen OOD rows in 128-row batches at concurrency 128:
#   full_ntp       full depth, stock classify head, NTP rows
#   full_prompted  full depth, the Gleipnir student-prompt workload
#   trunc_ntp      stopped after block 15, no probe head, NTP rows
#   trunc_probe    stopped after block 15, epoch-8 bank (110 probes), FP8 trip
#   trunc_single   stopped after block 15, the one deployed probe, raw input
# Re-running skips finished configurations. Usage: run_throughput.sh <name>
set -euo pipefail
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
SNAP=/workspace/gleipnir/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
NAME=$1
OUT=results/probe_optimized_serving/$NAME
LOGS=logs/runpod/probe_optimized/$NAME
mkdir -p "$OUT" "$LOGS"

serve() {  # label, start.py flags...
  local label=$1; shift
  setsid $PY -m experiments.probe_optimized_serving.start --snapshot $SNAP \
    --log "$LOGS/server_$label.log" "$@" > "$LOGS/launcher_$label.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:8010/health > /dev/null && return 0
    kill -0 $SERVER 2> /dev/null || { echo "server $label exited" >&2; return 1; }
    sleep 5
  done
  echo "server $label startup timeout" >&2
  return 1
}

stop() {
  kill -- -$SERVER 2> /dev/null || true
  for _ in $(seq 1 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || return 0
    sleep 2
  done
}

bench() {  # label, rendering
  $PY -m experiments.probe_optimized_serving.throughput --rendering "$2" \
    --snapshot $SNAP --label "$1" --out "$OUT/$1.json" > "$LOGS/bench_$1.log" 2>&1
  tail -1 "$LOGS/bench_$1.log"
}

if [ ! -f "$OUT/full_prompted.json" ]; then
  serve full
  bench full_ntp ntp
  bench full_prompted prompted
  stop
fi

if [ ! -f "$OUT/trunc_ntp.json" ]; then
  serve trunc --truncate
  bench trunc_ntp ntp
  stop
fi

if [ ! -f "$OUT/trunc_probe/timing.json" ]; then
serve probe --truncate --probe-bank results/probe_generalization/full01/ntp-fp8 \
  --epochs 8 --variants fp8
$PY -m experiments.probe_optimized_serving.probe_client \
  --rows data/probe_generalization/full/ood.jsonl --snapshot $SNAP \
  --receipt "$LOGS/server_probe.probe.json" --out "$OUT/trunc_probe" \
  > "$LOGS/bench_trunc_probe.log" 2>&1
tail -1 "$LOGS/bench_trunc_probe.log"
stop
fi

# The deployed scorer: the soft attention family's ID-best checkpoint and
# best-ID seed (user choice: attention probes are strongest in prior work).
# Raw activations: the FP8 round trip only matched the training store.
if [ ! -f "$OUT/trunc_single/timing.json" ]; then
serve single --truncate --probe-bank results/probe_generalization/full01/ntp-fp8 \
  --epochs 8 --variants raw --select attention-soft-s1
$PY -m experiments.probe_optimized_serving.probe_client \
  --rows data/probe_generalization/full/ood.jsonl --snapshot $SNAP \
  --receipt "$LOGS/server_single.probe.json" --out "$OUT/trunc_single" \
  > "$LOGS/bench_trunc_single.log" 2>&1
tail -1 "$LOGS/bench_trunc_single.log"
stop
fi
echo THROUGHPUT_DONE
