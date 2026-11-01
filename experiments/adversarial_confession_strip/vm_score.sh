#!/usr/bin/env bash
# On the A100 80GB VM `gleipnir-qwen-gpu`: update ~/gleipnir-coc from the
# staged bundle, unpack the stripped rows, check the FLA environment and the
# frozen inputs, then score with Gleipnir 4B v0.3 and 27B (canaries before and
# after each; resumable; no activations). Adapted from monitor-aware's
# vm_score.sh. Needs no secrets. Copy the staged files to ~ first:
#
#   bash ~/cs_vm_score.sh              # everything, in order
#   bash ~/cs_vm_score.sh 4b           # one monitor (4b or 27b)
#
# Progress and the outcome go to ~/cs_score_status.log ("ALL DONE" or
# "FAILED <step>"); full output to ~/gleipnir-coc/logs/gcp/adversarial_confession_strip/.
set -euo pipefail
status=~/cs_score_status.log
fail() { echo "$(date -u +%FT%TZ) FAILED $1" >> "$status"; exit 1; }
note() { echo "$(date -u +%FT%TZ) $1" >> "$status"; }
trap 'fail "${step:-setup}"' ERR

step=checksums
(cd ~ && shasum -a 256 -c CS_SHA256SUMS)
step=checkout
cd ~/gleipnir-coc
git fetch ~/cs_gleipnir.bundle feat/adversarial-stress-tests:refs/heads/cs-score-bundle --force
git checkout --detach cs-score-bundle
note "start at $(git rev-parse --short HEAD) on $(hostname)"
tar xzf ~/cs_score_data.tar.gz

step=environment
if [[ -f .v03-env ]]; then
    # shellcheck disable=SC1091
    source .v03-env
fi
export PY=${PY:-$PWD/.venv/bin/python}
export PYTHONPATH=${PYTHONPATH:-src:.}
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
if ! $PY -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f; assert f()" 2>/dev/null; then
    # The lab_v03 overlay lives in /tmp, which a VM restart clears: reinstall it.
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
config = json.load(open("experiments/adversarial_confession_strip/config.json"))
for key in ("rows_student", "rows_teacher", "canaries"):
    path = Path(config["data"]) / f"{key}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["sha256"][key], path
print("FLA available; inputs match their frozen SHA256")
PYEOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -m pytest -q -p no:cacheprovider experiments/adversarial_confession_strip/test_confession_strip.py -k scoring

logs=logs/gcp/adversarial_confession_strip
mkdir -p "$logs"
monitors=${1:-"4b 27b"}
for m in $monitors; do
    step="weights_$m"
    $PY - "$m" <<'PYEOF'
import json, sys
from huggingface_hub import snapshot_download
from gleipnir.adversarial.monitors import MONITORS
config = json.load(open("experiments/adversarial_confession_strip/config.json"))
spec = MONITORS[config["gpu_monitors"][sys.argv[1]]["monitor"]]
snapshot_download(spec.base, revision=spec.base_revision)
if spec.adapter:
    snapshot_download(spec.adapter, revision=spec.adapter_revision,
                      allow_patterns=["*.json", "*.txt", "adapter_model.safetensors"])
print("weights ready:", sys.argv[1])
PYEOF
    step="canary_$m"
    $PY experiments/adversarial_confession_strip/score.py canary --monitor "$m" 2>&1 | tee "$logs/canary_$m.log"
    note "canary $m passed"
    step="score_$m"
    $PY experiments/adversarial_confession_strip/score.py score --monitor "$m" 2>&1 | tee "$logs/score_$m.log"
    note "scored $m"
done
step=done
note "ALL DONE"
