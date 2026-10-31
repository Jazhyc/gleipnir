#!/usr/bin/env bash
# Prepare a fresh GPU VM (no preinstalled Python stack) for the lab runs.
# Run inside the copied checkout: bash experiments/adversarial_lab_v03/setup_vm.sh
# Locked environment (same packages as the A100 runs), FLA 0.5.2 overlay, base
# and v0.3 weights. Needs no secrets.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

./setup_dev.sh
uv pip install --python .venv/bin/python --no-deps --target /tmp/gleipnir-qwen35-fla \
    flash-linear-attention==0.5.2 fla-core==0.5.2

cat > .v03-env <<EOF
export PYTHONPATH=/tmp/gleipnir-qwen35-fla:src:.
export HF_HOME=$PWD/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export PY=$PWD/.venv/bin/python
EOF
# shellcheck disable=SC1091
source .v03-env

$PY - <<'EOF'
from huggingface_hub import snapshot_download
from transformers.utils.import_utils import is_flash_linear_attention_available

from gleipnir.adversarial.monitors import MONITORS

spec = MONITORS["4b-v03"]
snapshot_download(spec.base, revision=spec.base_revision)
snapshot_download(
    spec.adapter,
    revision=spec.adapter_revision,
    allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"],
)
assert is_flash_linear_attention_available(), "FLA overlay not importable"
print("weights ready; FLA available")
EOF
$PY -m pytest -q -p no:cacheprovider tests/adversarial/test_multiagent.py experiments/adversarial_lab_v03
