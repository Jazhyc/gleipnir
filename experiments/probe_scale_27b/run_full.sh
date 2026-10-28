#!/usr/bin/env bash
# Full 27B run on one Lambda GPU, the 4B full01 protocol at block 31 for one
# backbone (BACKBONE=ntp, the default, or prompted; set NAME per run): FP8
# extraction of train, ID and OOD to local disk, a background copy to the
# persistent filesystem, last-token probes (CPU) beside the fused pooled sweep
# (GPU), then the frozen ID selection (with the budget-extension rule) and the
# single OOD read. Re-running skips finished stages.
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
if [ -f "$HOME/.config/gleipnir/secrets.env" ]; then
  set -a; source "$HOME/.config/gleipnir/secrets.env"; set +a
fi
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_27b}
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores/qwen38_27b_l31}
BACKBONE=${BACKBONE:-ntp}
NAME=${NAME:-scale27b_full01}
LOGS=logs/lambda/probe_scale_27b/$NAME
RESULTS=results/probe_generalization/$NAME
mkdir -p "$LOGS" "$RESULTS"
PY=.venv/bin/python
LOCK=$LOGS/.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_full.sh holds $LOCK (pid $(cat "$LOCK/pid" 2>/dev/null))" >&2
  exit 75
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

store=$LOCAL/full/$BACKBONE
[ -f "$RESULTS/ood_$BACKBONE.json" ] && { echo "already done"; exit 0; }

for split in train id ood; do
  [ -f "$store/$split/meta.json" ] && continue
  rm -rf "${store:?}/$split"
  df -h "$HOME" | tail -1
  echo "extract_start $split $(date -Is)"
  $PY -m experiments.probe_generalization.extract --population full \
    --model qwen38_27b --layer 31 --backbone "$BACKBONE" --splits "$split" \
    --formats fp8 --out-root "$LOCAL" > "$LOGS/extract_$split.log" 2>&1
done
df -h "$HOME" | tail -1

echo "persist_start $(date -Is)"
mkdir -p "$PERSIST/full/$BACKBONE"
cp -r "$store/." "$PERSIST/full/$BACKBONE/" > "$LOGS/persist.log" 2>&1 &
persist_pid=$!

$PY -m experiments.probe_generalization.lasttoken --store-root "$LOCAL" \
  --population full --backbone "$BACKBONE" --name "$NAME" \
  > "$LOGS/lasttoken_$BACKBONE.log" 2>&1 &
lasttoken_pid=$!

if [ ! -f "$RESULTS/$BACKBONE-fp8/probes_e8.pt" ]; then
  rm -rf "$RESULTS/$BACKBONE-fp8"
  echo "train_start $(date -Is)"
  $PY -m experiments.probe_generalization.train --store-root "$LOCAL" \
    --population full --backbone "$BACKBONE" --train-format fp8 \
    --eval-formats fp8 --resident none --eval-resident none --seeds 0 1 2 3 4 \
    --checkpoints 1 2 4 8 --token-budget 65536 --name "$NAME" \
    > "$LOGS/train_$BACKBONE.log" 2>&1
fi
wait "$lasttoken_pid"

echo "ood_start $(date -Is)"
$PY -m experiments.probe_generalization.evaluate_ood --store-root "$LOCAL" \
  --population full --backbone "$BACKBONE" --name "$NAME" \
  > "$LOGS/ood_$BACKBONE.log" 2>&1

wait "$persist_pid"
diff <(cd "$store" && find . -type f -printf "%P %s\n" | sort) \
     <(cd "$PERSIST/full/$BACKBONE" && find . -type f -printf "%P %s\n" | sort) \
  > /dev/null || { echo "persist_mismatch" >&2; exit 1; }
echo FULL_DONE "$(date -Is)"
