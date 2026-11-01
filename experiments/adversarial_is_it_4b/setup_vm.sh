#!/usr/bin/env bash
# Prepare the H100 VM for this batch: the lab_v03 environment (locked packages,
# FLA 0.5.2 overlay) plus this batch's 4B checkpoints. The 27B is fetched by
# `fetch_27b` once the 4B arms are done. Needs no secrets.
#   bash experiments/adversarial_is_it_4b/setup_vm.sh          # env + 4B weights
#   bash experiments/adversarial_is_it_4b/setup_vm.sh fetch_27b
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

if [[ "${1:-}" != "fetch_27b" ]]; then
    if [[ ! -x .venv/bin/python ]]; then
        bash experiments/adversarial_lab_v03/setup_vm.sh
    else
        .venv/bin/python -c "import sys; sys.path.insert(0, '/tmp/gleipnir-qwen35-fla'); import fla" 2>/dev/null \
            || uv pip install --python .venv/bin/python --no-deps \
                --target /tmp/gleipnir-qwen35-fla flash-linear-attention==0.5.2 fla-core==0.5.2
        [[ -f .v03-env ]] || bash experiments/adversarial_lab_v03/setup_vm.sh
    fi
fi
# shellcheck disable=SC1091
source .v03-env
df -h . | tail -1

$PY - "${1:-4b}" <<'EOF'
import sys

from huggingface_hub import snapshot_download
from transformers.utils.import_utils import is_flash_linear_attention_available

from gleipnir.adversarial.monitors import MONITORS

keys = ["27b-merged"] if sys.argv[1] == "fetch_27b" else ["4b-v03-merged", "4b-base", "4b-ia40"]
for key in keys:
    spec = MONITORS[key]
    snapshot_download(spec.base, revision=spec.base_revision)
    if spec.adapter:
        snapshot_download(
            spec.adapter,
            revision=spec.adapter_revision,
            allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"],
        )
    print("ready:", key)
assert is_flash_linear_attention_available(), "FLA overlay not importable"
EOF
$PY -m pytest -q -p no:cacheprovider tests/adversarial/test_multiagent.py experiments/adversarial_is_it_4b
