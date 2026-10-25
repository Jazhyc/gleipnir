#!/usr/bin/env bash
# Full run on one Lambda GPU, one backbone at a time: FP8 extraction of train,
# ID and OOD to local disk, a background copy to the persistent filesystem,
# last-token probes (CPU) beside the fused pooled sweep (GPU), then the frozen
# ID selection and the single OOD read. The local store is removed only after
# its persistent copy matches file for file. Re-running skips finished stages.
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
if [ -f "$HOME/.config/gleipnir/secrets.env" ]; then
  set -a; source "$HOME/.config/gleipnir/secrets.env"; set +a
fi
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_full}
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores}
NAME=${NAME:-full01}
LOGS=logs/lambda/probe_generalization/$NAME
RESULTS=results/probe_generalization/$NAME
mkdir -p "$LOGS" "$RESULTS"
PY=.venv/bin/python
# One runner per campaign: a second launch exits instead of racing the first.
LOCK=$LOGS/.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_full.sh holds $LOCK (pid $(cat "$LOCK/pid" 2>/dev/null))" >&2
  exit 75
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

same_files() {  # dir_a dir_b: identical relative paths and sizes
  diff <(cd "$1" && find . -type f -printf "%P %s\n" | sort) \
       <(cd "$2" && find . -type f -printf "%P %s\n" | sort) > /dev/null
}

for backbone in prompted ntp; do
  store=$LOCAL/full/$backbone
  done_flag=$RESULTS/ood_$backbone.json
  [ -f "$done_flag" ] && { echo "skip $backbone (done)"; continue; }

  todo=()
  for split in train id ood; do
    [ -f "$store/$split/meta.json" ] || todo+=("$split")
  done
  if [ ${#todo[@]} -gt 0 ]; then
    for split in "${todo[@]}"; do rm -rf "${store:?}/$split"; done
    echo "extract_start $backbone ${todo[*]} $(date -Is)"
    $PY -m experiments.probe_generalization.extract --population full \
      --backbone "$backbone" --splits "${todo[@]}" --formats fp8 \
      --out-root "$LOCAL" > "$LOGS/extract_$backbone.log" 2>&1
  fi

  echo "persist_start $backbone $(date -Is)"
  mkdir -p "$PERSIST/full/$backbone"
  cp -r "$store/." "$PERSIST/full/$backbone/" > "$LOGS/persist_$backbone.log" 2>&1 &
  persist_pid=$!

  $PY -m experiments.probe_generalization.lasttoken --store-root "$LOCAL" \
    --population full --backbone "$backbone" --name "$NAME" \
    > "$LOGS/lasttoken_$backbone.log" 2>&1 &
  lasttoken_pid=$!

  if [ ! -f "$RESULTS/$backbone-fp8/probes_e8.pt" ]; then
    rm -rf "$RESULTS/$backbone-fp8"
    echo "train_start $backbone $(date -Is)"
    $PY -m experiments.probe_generalization.train --store-root "$LOCAL" \
      --population full --backbone "$backbone" --train-format fp8 \
      --eval-formats fp8 --resident none --seeds 0 1 2 3 4 \
      --checkpoints 1 2 4 8 --token-budget 65536 --name "$NAME" \
      > "$LOGS/train_$backbone.log" 2>&1
  fi
  wait "$lasttoken_pid"

  echo "ood_start $backbone $(date -Is)"
  $PY -m experiments.probe_generalization.evaluate_ood --store-root "$LOCAL" \
    --population full --backbone "$backbone" --name "$NAME" \
    > "$LOGS/ood_$backbone.log" 2>&1

  wait "$persist_pid"
  if same_files "$store" "$PERSIST/full/$backbone"; then
    echo "persisted $backbone; removing local store $(date -Is)"
    rm -rf "$store"
  else
    echo "persist_mismatch $backbone; keeping local store" >&2
    exit 1
  fi
done
echo FULL_DONE "$(date -Is)"
