#!/usr/bin/env bash
# Pilot on one Lambda GPU: extraction for both backbones (BF16 + FP8 stores),
# parity, the BF16/FP8 probe sweeps, last-token probes, then the frozen gates.
# STORES may point at a local-disk copy of the persistent stores. Re-running skips stages whose outputs exist.
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
if [ -f "$HOME/.config/gleipnir/secrets.env" ]; then
  set -a; source "$HOME/.config/gleipnir/secrets.env"; set +a
fi
export PYTHONPATH=src:.
STORES=${STORES:-/lambda/nfs/gleipnir-probes/stores}
NAME=${NAME:-pilot01}
LOGS=logs/lambda/probe_generalization/$NAME
mkdir -p "$LOGS"
PY=.venv/bin/python

extract() {  # backbone split...
  local backbone=$1; shift
  local todo=()
  for split in "$@"; do
    [ -f "$STORES/pilot/$backbone/$split/meta.json" ] || todo+=("$split")
  done
  [ ${#todo[@]} -eq 0 ] && return
  for split in "${todo[@]}"; do rm -rf "$STORES/pilot/$backbone/$split"; done
  echo "extract_start $backbone ${todo[*]} $(date -Is)"
  $PY -m experiments.probe_generalization.extract --population pilot \
    --backbone "$backbone" --splits "${todo[@]}" --formats bf16 fp8 \
    --out-root "$STORES" > "$LOGS/extract_$backbone.log" 2>&1
}

extract prompted parity train id
extract ntp train id

# Last-token probes are CPU-only; run them alongside the GPU sweeps.
lasttoken_pids=()
for backbone in prompted ntp; do
  $PY -m experiments.probe_generalization.lasttoken --store-root "$STORES" \
    --backbone "$backbone" --name "$NAME" > "$LOGS/lasttoken_$backbone.log" 2>&1 &
  lasttoken_pids+=($!)
done

# One fused sweep at a time: FP8 rows resident on the GPU, BF16 rows (twice
# the bytes) in pinned CPU memory.
for backbone in prompted ntp; do
  for fmt in fp8 bf16; do
    if [ ! -f "results/probe_generalization/$NAME/$backbone-$fmt/probes_e16.pt" ]; then
      rm -rf "results/probe_generalization/$NAME/$backbone-$fmt"
      resident=$([ "$fmt" = fp8 ] && echo gpu || echo cpu)
      echo "train_start $backbone $fmt resident=$resident $(date -Is)"
      $PY -m experiments.probe_generalization.train --store-root "$STORES" \
        --backbone "$backbone" --train-format "$fmt" --resident "$resident" \
        --name "$NAME" > "$LOGS/train_${backbone}_$fmt.log" 2>&1
    fi
  done
done
for pid in "${lasttoken_pids[@]}"; do wait "$pid"; done

$PY -m experiments.probe_generalization.pilot --store-root "$STORES" --name "$NAME" \
  | tee "$LOGS/pilot.log"
echo PILOT_DONE "$(date -Is)"
