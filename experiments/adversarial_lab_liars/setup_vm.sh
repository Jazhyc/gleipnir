#!/usr/bin/env bash
# Prepare an H100 80 GB VM for this study (modelled on adversarial_is_it_4b):
# the lab_v03 environment (locked packages, FLA 0.5.2 overlay, base and v0.3
# 4B weights), then the 27B merged checkpoint on request. Needs no secrets.
#   bash experiments/adversarial_lab_liars/setup_vm.sh            # env + 4B
#   bash experiments/adversarial_lab_liars/setup_vm.sh fetch_27b  # 27B (~54 GB)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

if [[ "${1:-}" != "fetch_27b" ]]; then
    if [[ ! -x .venv/bin/python || ! -f .v03-env ]]; then
        bash experiments/adversarial_lab_v03/setup_vm.sh
    fi
fi
# shellcheck disable=SC1091
source .v03-env
df -h . | tail -1

$PY - "${1:-4b}" <<'EOF'
import json
import sys

from huggingface_hub import snapshot_download
from transformers.utils.import_utils import is_flash_linear_attention_available

from gleipnir.adversarial.monitors import MONITORS

config = json.load(open("experiments/adversarial_lab_liars/config.json"))
key = "27b" if sys.argv[1] == "fetch_27b" else "4b"
spec = MONITORS[config["monitors"][key]["monitor"]]
snapshot_download(spec.base, revision=spec.base_revision)
if spec.adapter:
    snapshot_download(
        spec.adapter,
        revision=spec.adapter_revision,
        allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"],
    )
assert is_flash_linear_attention_available(), "FLA overlay not importable"
print("ready:", key, config["monitors"][key]["monitor"])
EOF

# The frozen inputs must be present and match config.json before any scoring.
$PY - <<'EOF'
import json
from pathlib import Path

from gleipnir.adversarial.eval_sets import file_sha256

config = json.load(open("experiments/adversarial_lab_liars/config.json"))
data = Path(config["data"])
for name, expected in config["sha256"].items():
    path = data / (f"{name}.json" if name == "kimi_order" else f"{name}.jsonl")
    assert file_sha256(path) == expected, f"{path} does not match its frozen SHA256"
print("inputs match their frozen SHA256")
EOF
$PY -m pytest -q -p no:cacheprovider tests/adversarial/test_multiagent.py experiments/adversarial_lab_liars
