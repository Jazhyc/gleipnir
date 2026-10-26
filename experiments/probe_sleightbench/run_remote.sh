#!/usr/bin/env bash
# One Lambda GPU with the bootstrapped project venv: NTP FP8 extraction of the
# parity and SleightBench rows at full length, then the parity gate and the
# single scoring pass. The picked probes and their stored ID logits are pushed
# with the rows; the activation store is copied to the persistent filesystem. Metrics run locally
# (`score.py report`). Re-running skips a finished extraction; scoring refuses
# to repeat.
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
export PYTHONPATH=src:.
STORES=${STORES:-$HOME/stores_sb}
NAME=${NAME:-sbprobe01}
LOGS=logs/lambda/probe_sleightbench/$NAME
mkdir -p "$LOGS"
PY=.venv/bin/python

store=$STORES/sleightbench/ntp
if [ ! -f "$store/stripped/meta.json" ]; then
  rm -rf "$store"
  echo "extract_start $(date -Is)"
  # Parity rows first: a failed environment stops before SleightBench rows.
  $PY -m experiments.probe_generalization.extract --population sleightbench \
    --backbone ntp --splits parity_id stripped --formats fp8 \
    --out-root "$STORES" > "$LOGS/extract.log" 2>&1
fi
echo "score_start $(date -Is)"
$PY -m experiments.probe_sleightbench.score score --store-root "$STORES" \
  --name "$NAME" 2>&1 | tee "$LOGS/score.log"
cp "$store/parity_id/index.jsonl" "$LOGS/parity_index.jsonl"
cp "$store/stripped/index.jsonl" "$LOGS/stripped_index.jsonl"
cp "$store/stripped/meta.json" "$LOGS/stripped_meta.json"
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores}
mkdir -p "$PERSIST/sleightbench/ntp"
cp -r "$store/." "$PERSIST/sleightbench/ntp/"
diff <(cd "$store" && find . -type f -printf "%P %s\n" | sort) \
     <(cd "$PERSIST/sleightbench/ntp" && find . -type f -printf "%P %s\n" | sort)
echo "persisted $store $(date -Is)"
echo SB_PROBE_DONE "$(date -Is)"
