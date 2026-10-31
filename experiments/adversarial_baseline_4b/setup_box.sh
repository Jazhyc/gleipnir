#!/usr/bin/env bash
# Prepare a fresh single-GPU box (e.g. JarvisLabs) for adversarial scoring:
#   git clone -b feat/adversarial-stress-tests https://github.com/Jazhyc/gleipnir.git
#   cd gleipnir && bash experiments/adversarial_baseline_4b/setup_box.sh 4b
# Locked environment, the pinned FLA 0.5.2 overlay (as in docs/infrastructure.md),
# APPS at its pinned revision, rebuilt ID/OOD sets and one monitor's weights.
# Needs no secrets. causal-conv1d is skipped; Transformers uses its exact torch
# fallback for the short convolution.
set -euo pipefail
monitor="${1:-4b}"
cd "$(dirname "$0")/../.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

./setup_dev.sh
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/.cache/huggingface"

fla=/tmp/gleipnir-qwen35-fla
if [ ! -d "$fla/fla" ]; then
    uv pip install --python .venv/bin/python --no-deps --target "$fla" \
        flash-linear-attention==0.5.2 fla-core==0.5.2
fi
echo "export PYTHONPATH=$fla:\${PYTHONPATH:-}" > .adversarial-env
echo "export HF_HOME=$HF_HOME HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false" \
    >> .adversarial-env
# shellcheck disable=SC1091
source .adversarial-env

.venv/bin/python - "$monitor" <<'EOF'
import sys

from huggingface_hub import snapshot_download

from gleipnir.adversarial.eval_sets import APPS_REPO, APPS_REVISION
from gleipnir.adversarial.monitors import MONITORS

snapshot_download(
    APPS_REPO,
    repo_type="dataset",
    revision=APPS_REVISION,
    allow_patterns=["data/*"],
    local_dir=".",
)
spec = MONITORS[sys.argv[1]]
snapshot_download(spec.base, revision=spec.base_revision)
snapshot_download(
    spec.adapter,
    revision=spec.adapter_revision,
    allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"],
)
EOF

PYTHONPATH="src:.:$PYTHONPATH" .venv/bin/python scripts/build_eval_datasets.py all
.venv/bin/python -c "
from transformers.utils.import_utils import is_flash_linear_attention_available as ok
assert ok(), 'FLA overlay not importable'
print('FLA available')"
.venv/bin/python -m pytest -q tests/adversarial experiments/adversarial_baseline_4b
echo "Ready. Run: source .adversarial-env && .venv/bin/python experiments/adversarial_baseline_4b/run.py smoke"
