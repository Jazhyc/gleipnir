#!/usr/bin/env bash
# The 27B follow-up on one B200 (README "27B follow-up"). Stages, each skipped
# once its outputs exist:
#   prepare     render/tokenize rows and keep tables (prepare_27b.py)
#   diag        BF16 Transformers extraction of the 64 diagnostic rows
#   quality     per backbone: truncated server with the attention picks' banks
#               (fp8, raw); ID, OOD and diagnostic rows; diagnostic states dumped
#   diagnose    activation stats and the plumbing check (diag27b.py)
#   throughput  fresh compiled servers: full depth, truncated, deployed probe
# Assumes the restored serving stack and the base snapshot under /root/hf-27b.
set -euo pipefail
cd /workspace/gleipnir
export PYTHONPATH=src:.
PY=/tmp/gleipnir-vllm031-runtime/bin/python
TPY=.venv/bin/python
export HF_HOME=/root/hf-27b
SNAP=$HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
ROWS=data/probe_optimized_serving/qwen38_27b
OPT=results/probe_optimized_serving/opt27b01
THR=results/probe_optimized_serving/thr27b01
LOGS=logs/runpod/probe_optimized_27b
STORE=/root/diag27b
mkdir -p "$OPT" "$THR" "$LOGS"
declare -A SWEEP=([ntp]=results/probe_generalization/scale27b_full01
                  [prompted]=results/probe_generalization/scale27b_prompted01)
declare -A EPOCHS=([ntp]=2 [prompted]=12,1)
declare -A DEPLOYED=([ntp]="2 attention-soft-s1" [prompted]="12 attention-soft-s3")

serve() {  # label, start.py flags...; retries the MXFP8 compile race
  local label=$1; shift
  for attempt in 1 2 3; do
    setsid $PY -m experiments.probe_optimized_serving.start --target 27b \
      --snapshot "$SNAP" --log "$LOGS/server_$label.log" "$@" \
      > "$LOGS/launcher_$label.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 480); do
      curl -sf http://127.0.0.1:8010/health > /dev/null && return 0
      kill -0 $SERVER 2> /dev/null || break
      sleep 5
    done
    if kill -0 $SERVER 2> /dev/null; then
      echo "server $label startup timeout" >&2; stop; return 1
    fi
    if grep -q TYPE_UNSTABLE_JOIN "$LOGS/server_$label.log"; then
      cp "$LOGS/server_$label.log" "$LOGS/server_${label}.failed_compile_race$attempt.log"
      echo "server $label: MXFP8 compile race, retry $attempt" >&2
      continue
    fi
    echo "server $label exited" >&2; return 1
  done
  return 1
}

stop() {
  kill -- -$SERVER 2> /dev/null || true
  for _ in $(seq 1 90); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || return 0
    sleep 2
  done
}

client() {  # rows file, receipt label, out dir, log
  $PY -m experiments.probe_optimized_serving.probe_client --rows "$1" \
    --snapshot "$SNAP" --receipt "$LOGS/server_$2.probe.json" --out "$3" > "$4" 2>&1
  tail -1 "$4"
}

if [ ! -f "$ROWS/keep_prompted.json" ]; then
  $TPY -m experiments.probe_optimized_serving.prepare_27b > "$LOGS/prepare.log" 2>&1
  tail -6 "$LOGS/prepare.log"
fi

for b in ntp prompted; do
  if [ ! -f "$STORE/optimized_diag/$b/ood/meta.json" ]; then
    $TPY -m experiments.probe_generalization.extract --population optimized_diag \
      --model qwen38_27b --backbone $b --splits ood --formats bf16 fp8 --layer 31 \
      --out-root "$STORE" > "$LOGS/diag_bf16_$b.log" 2>&1
    tail -1 "$LOGS/diag_bf16_$b.log"
  fi
done

for b in ntp prompted; do
  if [ ! -f "$OPT/$b/ood/timing.json" ]; then
    dump=$OPT/$b/dump
    mkdir -p "$dump"
    jq -sc 'map(.key)' "$ROWS/${b}_diag.jsonl" > "$dump/keys.json"
    keep=()
    [ $b = prompted ] && keep=(--keep "$ROWS/keep_prompted.json")
    serve quality_$b --truncate --probe-bank "${SWEEP[$b]}/$b-fp8" \
      --epochs "${EPOCHS[$b]}" --variants fp8,raw --dump "$dump" "${keep[@]}"
    for split in diag id ood; do
      client "$ROWS/${b}_$split.jsonl" quality_$b "$OPT/$b/$split" \
        "$LOGS/quality_${b}_$split.log"
    done
    stop
  fi
done

for b in ntp prompted; do
  if [ ! -f "$OPT/$b/diag.json" ]; then
    $TPY -m experiments.probe_optimized_serving.diag27b --backbone $b \
      --store "$STORE/optimized_diag/$b/ood" --dump "$OPT/$b/dump" \
      --rows "$ROWS/${b}_diag.jsonl" --sweep "${SWEEP[$b]}" \
      --out "$OPT/$b/diag.json" > "$LOGS/diagnose_$b.log" 2>&1
    tail -3 "$LOGS/diagnose_$b.log"
  fi
done

bench() {  # label, backbone
  $PY -m experiments.probe_optimized_serving.throughput --rendering ntp \
    --rows "$ROWS/${2}_ood.jsonl" --snapshot "$SNAP" --label "$1" \
    --out "$THR/$1.json" > "$LOGS/bench_$1.log" 2>&1
  tail -1 "$LOGS/bench_$1.log"
}

if [ ! -f "$THR/full_prompted.json" ]; then
  serve full
  bench full_ntp ntp
  bench full_prompted prompted
  stop
fi

if [ ! -f "$THR/trunc_prompted.json" ]; then
  serve trunc --truncate
  bench trunc_ntp ntp
  bench trunc_prompted prompted
  stop
fi

# The deployed scorer per backbone: the soft attention pick's best-ID seed,
# raw activations (the 4B rule).
for b in ntp prompted; do
  if [ ! -f "$THR/single_$b/timing.json" ]; then
    read -r epoch probe <<< "${DEPLOYED[$b]}"
    keep=()
    [ $b = prompted ] && keep=(--keep "$ROWS/keep_prompted.json")
    serve single_$b --truncate --probe-bank "${SWEEP[$b]}/$b-fp8" --epochs "$epoch" \
      --variants raw --select "$probe" "${keep[@]}"
    client "$ROWS/${b}_ood.jsonl" single_$b "$THR/single_$b" "$LOGS/bench_single_$b.log"
    stop
  fi
done
echo RUN27B_DONE
