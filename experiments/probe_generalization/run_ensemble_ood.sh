#!/usr/bin/env bash
# Post-hoc OOD read of the ID-selected seed ensembles on one Lambda GPU attached
# to the gleipnir-probes filesystem. Scoring needs only Torch, NumPy, pandas
# and scikit-learn, so a light venv with the locked versions replaces the full
# bootstrap (the driver-570 image lacks the locked CUDA-13 runtime). Each
# backbone's OOD store is staged to local disk (NFS page faults stall).
set -euo pipefail
cd "$HOME/gleipnir"
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_full}
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores}
NAME=${NAME:-full01}
LOGS=logs/lambda/probe_generalization/$NAME
mkdir -p "$LOGS"
PY=$HOME/venv-score/bin/python

if [ ! -x "$PY" ]; then
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 "$HOME/venv-score"
  uv pip install --python "$PY" torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$PY" numpy==2.3.5 pandas==3.0.5 scikit-learn==1.9.0
fi
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"

for backbone in prompted ntp; do
  store=$LOCAL/full/$backbone/ood
  if [ ! -f "$store/meta.json" ]; then
    echo "stage_start $backbone $(date -Is)"
    mkdir -p "$store"
    cp -r "$PERSIST/full/$backbone/ood/." "$store/"
  fi
  echo "score_start $backbone $(date -Is)"
  $PY -m experiments.probe_generalization.ensemble_ood --store-root "$LOCAL" \
    --population full --backbone "$backbone" --name "$NAME" \
    2>&1 | tee "$LOGS/ensemble_ood_$backbone.log"
done
echo ENSEMBLE_OOD_DONE "$(date -Is)"
