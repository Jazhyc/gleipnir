#!/usr/bin/env bash
# On the A100 80GB VM `gleipnir-qwen-gpu`: update ~/gleipnir-coc from the
# staged bundle, unpack arm 1's rows, check FLA and the frozen inputs, run this
# study's tests, then score with Gleipnir 4B v0.3 and 27B (monitor-aware's
# canaries before and after each; decision-token activations saved for every
# row; resumable). Needs no secrets. Copy the staged files to ~ first:
#
#   bash ~/vm_score.sh            # 4b then 27b
#   bash ~/vm_score.sh 27b        # one monitor
#
# Status: ~/qc_score_status.log ("ALL DONE" or "FAILED <step>"); full output
# in ~/gleipnir-coc/logs/gcp/adversarial_quiet_cheats/. Stop at 1 hour.
set -euo pipefail
status=~/qc_score_status.log
fail() { echo "$(date -u +%FT%TZ) FAILED $1" >> "$status"; exit 1; }
note() { echo "$(date -u +%FT%TZ) $1" >> "$status"; }
trap 'fail "${step:-setup}"' ERR

step=checksums
(cd ~ && shasum -a 256 -c SHA256SUMS)
step=checkout
cd ~/gleipnir-coc
git fetch ~/gleipnir.bundle feat/adversarial-stress-tests:refs/heads/qc-score-bundle --force
git checkout --detach qc-score-bundle
note "start at $(git rev-parse --short HEAD) on $(hostname)"
tar xzf ~/qc_score_data.tar.gz

step=environment
if [[ -f .v03-env ]]; then
    # shellcheck disable=SC1091
    source .v03-env
fi
export PY=${PY:-$PWD/.venv/bin/python}
export PYTHONPATH=${PYTHONPATH:-src:.}
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
if ! $PY -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f; assert f()" 2>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    uv pip install --python .venv/bin/python --no-deps --target /tmp/gleipnir-qwen35-fla \
        flash-linear-attention==0.5.2 fla-core==0.5.2
    export PYTHONPATH=/tmp/gleipnir-qwen35-fla:src:.
fi
$PY - <<'PYEOF'
import hashlib, json
from pathlib import Path
from transformers.utils.import_utils import is_flash_linear_attention_available
assert is_flash_linear_attention_available(), "FLA not importable"
arm1 = json.load(open("experiments/adversarial_quiet_cheats/config.json"))["arm1_deletion"]
ma = json.load(open("experiments/adversarial_monitor_aware/config.json"))["scoring"]
for key in ("rows_student", "rows_teacher"):
    path = Path("data/adversarial_quiet_cheats/arm1") / f"{key}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == arm1["rows_sha256"][key], path
path = Path(ma["data"]) / "canaries.jsonl"
assert hashlib.sha256(path.read_bytes()).hexdigest() == ma["sha256"]["canaries"], path
print("FLA available; inputs match their frozen SHA256")
PYEOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -m pytest -q -p no:cacheprovider experiments/adversarial_quiet_cheats/test_quiet_cheats.py

logs=logs/gcp/adversarial_quiet_cheats
mkdir -p "$logs"
monitors=${1:-"4b 27b"}
for m in $monitors; do
    step="weights_$m"
    $PY - "$m" <<'PYEOF'
import json, sys
from huggingface_hub import snapshot_download
from gleipnir.adversarial.monitors import MONITORS
config = json.load(open("experiments/adversarial_monitor_aware/config.json"))["scoring"]
spec = MONITORS[config["gpu_monitors"][sys.argv[1]]["monitor"]]
snapshot_download(spec.base, revision=spec.base_revision)
if spec.adapter:
    snapshot_download(spec.adapter, revision=spec.adapter_revision,
                      allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"])
print("weights ready:", sys.argv[1])
PYEOF
    step="score_$m"
    $PY experiments/adversarial_quiet_cheats/score.py score --monitor "$m" 2>&1 | tee "$logs/score_$m.log"
    note "scored $m"
done
step=done
note "ALL DONE"
