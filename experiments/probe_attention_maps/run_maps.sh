#!/usr/bin/env bash
# Attention-map pass over the NTP OOD store on one Lambda GPU attached to the
# gleipnir-probes filesystem. Light venv as in run_ensemble_ood.sh; the store
# is staged to local disk (NFS page faults stall).
set -euo pipefail
cd "$HOME/gleipnir"
export PYTHONPATH=src:.
LOCAL=${LOCAL:-$HOME/stores_full}
PERSIST=${PERSIST:-/lambda/nfs/gleipnir-probes/stores}
NAME=${NAME:-maps01}
LOGS=logs/lambda/probe_attention_maps/$NAME
mkdir -p "$LOGS"
PY=$HOME/venv-maps/bin/python
export PATH=$HOME/.local/bin:$PATH

if [ ! -x "$PY" ]; then
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 "$HOME/venv-maps"
  uv pip install --python "$PY" torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu129
  uv pip install --python "$PY" numpy==2.3.5 tokenizers==0.22.2 \
    huggingface-hub==1.33.0
fi
# The extraction's tokenizer class (added after the first launch).
$PY -c "import transformers" 2> /dev/null ||
  uv pip install --python "$PY" transformers==5.14.1
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"

store=$LOCAL/full/ntp/ood
if [ ! -f "$store/meta.json" ]; then
  echo "stage_start $(date -Is)"
  mkdir -p "$store"
  cp -r "$PERSIST/full/ntp/ood/." "$store/"
fi
echo "maps_start $(date -Is)"
$PY -m experiments.probe_attention_maps.analyze --store-root "$LOCAL" \
  --name "$NAME" 2>&1 | tee "$LOGS/analyze.log"
echo MAPS_DONE "$(date -Is)"
