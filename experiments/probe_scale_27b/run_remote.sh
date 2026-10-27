#!/usr/bin/env bash
# Scale pilot on one Lambda GPU: FP8 NTP extraction of the 27B (block 31) on
# train/ID/OOD-sample and of the 4B (block 15) on the OOD sample, a background
# copy to the persistent filesystem, the 27B pilot sweep (the 4B pilot01
# protocol), then the frozen ID selection and one OOD read for each backbone.
# Re-running skips finished stages.
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
if [ -f "$HOME/.config/gleipnir/secrets.env" ]; then
  set -a; source "$HOME/.config/gleipnir/secrets.env"; set +a
fi
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_scale}
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores/scale_pilot}
NAME=${NAME:-scale27b_pilot01}
LOGS=logs/lambda/probe_scale_27b/$NAME
RESULTS=results/probe_scale_27b/$NAME
mkdir -p "$LOGS" "$RESULTS"
PY=.venv/bin/python
SWEEP=results/probe_generalization/$NAME/ntp-fp8
PILOT4B=results/probe_generalization/pilot01/ntp-fp8

extract() {  # model layer split...
  local model=$1 layer=$2; shift 2
  local root=$LOCAL/$model todo=()
  for split in "$@"; do
    [ -f "$root/scale_pilot/ntp/$split/meta.json" ] || todo+=("$split")
  done
  [ ${#todo[@]} -eq 0 ] && return
  for split in "${todo[@]}"; do rm -rf "${root:?}/scale_pilot/ntp/$split"; done
  echo "extract_start $model ${todo[*]} $(date -Is)"
  $PY -m experiments.probe_generalization.extract --population scale_pilot \
    --model "$model" --layer "$layer" --backbone ntp --splits "${todo[@]}" \
    --formats fp8 --out-root "$root" > "$LOGS/extract_$model.log" 2>&1
}

extract qwen35_4b 15 ood
extract qwen38_27b 31 train id ood

echo "persist_start $(date -Is)"
mkdir -p "$PERSIST"
cp -r "$LOCAL/." "$PERSIST/" > "$LOGS/persist.log" 2>&1 &
persist_pid=$!

if [ ! -f "$SWEEP/probes_e16.pt" ]; then
  rm -rf "$SWEEP"
  echo "train_start $(date -Is)"
  $PY -m experiments.probe_generalization.train \
    --store-root "$LOCAL/qwen38_27b" --population scale_pilot --backbone ntp \
    --train-format fp8 --eval-formats fp8 \
    --architectures mean_linear mean_mlp attention multimax rolling_w10 \
      rolling_w64 rolling_w256 \
    --objectives hard soft --seeds 0 1 2 --checkpoints 2 4 8 16 --batch 16 \
    --resident cpu --token-budget 65536 --name "$NAME" \
    > "$LOGS/train.log" 2>&1
fi

for model in qwen38_27b qwen35_4b; do
  [ -f "$RESULTS/$model/ood.json" ] && continue
  rm -rf "${RESULTS:?}/$model"
  run=$([ "$model" = qwen38_27b ] && echo "$SWEEP" || echo "$PILOT4B")
  echo "ood_start $model $(date -Is)"
  $PY -m experiments.probe_scale_27b.score --run "$run" \
    --ood-store "$LOCAL/$model/scale_pilot/ntp/ood" --out "$RESULTS/$model" \
    > "$LOGS/ood_$model.log" 2>&1
done

wait "$persist_pid"
diff <(cd "$LOCAL" && find . -type f -printf "%P %s\n" | sort) \
     <(cd "$PERSIST" && find . -type f -printf "%P %s\n" | sort) > /dev/null \
  || { echo "persist_mismatch" >&2; exit 1; }
echo SCALE_PILOT_DONE "$(date -Is)"
