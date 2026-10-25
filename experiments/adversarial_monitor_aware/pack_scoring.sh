#!/usr/bin/env bash
# Local (Mac): stage what the GPU VM needs for monitor scoring (Amendment 4) in
# one folder, for `gcloud compute scp`. Run after committing:
#
#   experiments/adversarial_monitor_aware/pack_scoring.sh [STAGE_DIR]
#
# Stages:
#   gleipnir.bundle        git bundle of feat/adversarial-stress-tests (HEAD)
#   ma_score_data.tar.gz   data/adversarial_monitor_aware/{rows_student,rows_teacher,canaries}.jsonl
#                          and manifest.json (paths relative to the repository root)
#   vm_score.sh            the VM script (also inside the bundle)
#   SHA256SUMS
# No secrets are staged; GPU scoring needs none.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
stage=${1:-$repo/../../cache/ma_score_stage}
branch=feat/adversarial-stress-tests
data=data/adversarial_monitor_aware

[ "$(git -C "$repo" branch --show-current)" = "$branch" ] || { echo "not on $branch"; exit 1; }
git -C "$repo" diff --quiet HEAD -- experiments/adversarial_monitor_aware \
    || { echo "uncommitted study changes; commit first"; exit 1; }
(cd "$repo" && PYTHONPATH=src:. python3 - <<'EOF'
import hashlib, json
from pathlib import Path
config = json.load(open("experiments/adversarial_monitor_aware/config.json"))["scoring"]
for key in ("rows_student", "rows_teacher", "canaries"):
    path = Path(config["data"]) / f"{key}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["sha256"][key], path
print("inputs match their frozen SHA256")
EOF
)
mkdir -p "$stage"
rm -f "$stage"/gleipnir.bundle "$stage"/ma_score_data.tar.gz "$stage"/SHA256SUMS
git -C "$repo" bundle create "$stage/gleipnir.bundle" "$branch"
(cd "$repo" && COPYFILE_DISABLE=1 tar czf "$stage/ma_score_data.tar.gz" \
    "$data"/rows_student.jsonl "$data"/rows_teacher.jsonl "$data"/canaries.jsonl "$data"/manifest.json)
cp "$repo/experiments/adversarial_monitor_aware/vm_score.sh" "$stage/"
(cd "$stage" && shasum -a 256 gleipnir.bundle ma_score_data.tar.gz vm_score.sh > SHA256SUMS)
echo "staged in $stage at $(git -C "$repo" rev-parse --short HEAD):"
ls -l "$stage"
