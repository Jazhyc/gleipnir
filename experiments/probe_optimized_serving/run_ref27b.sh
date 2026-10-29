#!/usr/bin/env bash
# BF16 per-row OOD references for the 27B follow-up: ensemble_ood.py rescoring
# of the stored block-31 FP8 OOD activations with every probe and checkpoint,
# on one Lambda GPU attached to gleipnir-probes. Same light venv and local
# staging as probe_generalization/run_ensemble_ood.sh.
set -euo pipefail
cd "$HOME/gleipnir"
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_27b}
PERSIST=/lambda/nfs/gleipnir-probes/stores/qwen38_27b_l31
LOGS=logs/lambda/probe_optimized_27b
mkdir -p "$LOGS"
PY=$HOME/venv-score/bin/python
declare -A NAME=([ntp]=scale27b_full01 [prompted]=scale27b_prompted01)

if [ ! -x "$PY" ]; then
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 "$HOME/venv-score"
  uv pip install --python "$PY" torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu129
  uv pip install --python "$PY" numpy==2.3.5 pandas==3.0.5 scikit-learn==1.9.0
fi
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"

for backbone in ntp prompted; do
  name=${NAME[$backbone]}
  [ -f "results/probe_generalization/$name/ensemble_ood/logits_$backbone.npz" ] && continue
  store=$LOCAL/full/$backbone/ood
  if [ ! -f "$store/meta.json" ]; then
    echo "stage_start $backbone $(date -Is)"
    mkdir -p "$store"
    cp -r "$PERSIST/full/$backbone/ood/." "$store/"
  fi
  echo "score_start $backbone $(date -Is)"
  $PY -m experiments.probe_generalization.ensemble_ood --store-root "$LOCAL" \
    --population full --backbone "$backbone" --name "$name" \
    2>&1 | tee "$LOGS/ensemble_ood_$backbone.log"
  rm -rf "$store"
done
echo REF27B_DONE "$(date -Is)"
