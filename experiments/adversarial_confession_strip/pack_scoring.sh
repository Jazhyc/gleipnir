#!/usr/bin/env bash
# Local (Mac): stage what the GPU VM needs in one folder, for `gcloud compute
# scp`. Run after committing (adapted from monitor-aware's pack_scoring.sh):
#
#   experiments/adversarial_confession_strip/pack_scoring.sh [STAGE_DIR]
#
# Stages (names prefixed cs_ so they can sit next to other studies' files in ~):
#   cs_gleipnir.bundle     git bundle of feat/adversarial-stress-tests (HEAD)
#   cs_score_data.tar.gz   data/adversarial_confession_strip/{rows_student,rows_teacher,canaries}.jsonl
#                          and manifest.json (paths relative to the repository root)
#   cs_vm_score.sh         the VM script (also inside the bundle)
#   CS_SHA256SUMS
# No secrets are staged; GPU scoring needs none.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
stage=${1:-$repo/../../cache/cs_score_stage}
branch=feat/adversarial-stress-tests
data=data/adversarial_confession_strip

[ "$(git -C "$repo" branch --show-current)" = "$branch" ] || { echo "not on $branch"; exit 1; }
git -C "$repo" diff --quiet HEAD -- experiments/adversarial_confession_strip \
    || { echo "uncommitted study changes; commit first"; exit 1; }
(cd "$repo" && python3 - <<'PYEOF'
import hashlib, json
from pathlib import Path
config = json.load(open("experiments/adversarial_confession_strip/config.json"))
for key in ("rows_student", "rows_teacher", "canaries"):
    path = Path(config["data"]) / f"{key}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["sha256"][key], path
print("inputs match their frozen SHA256")
PYEOF
)
mkdir -p "$stage"
rm -f "$stage"/cs_gleipnir.bundle "$stage"/cs_score_data.tar.gz "$stage"/CS_SHA256SUMS
git -C "$repo" bundle create "$stage/cs_gleipnir.bundle" "$branch"
(cd "$repo" && COPYFILE_DISABLE=1 tar czf "$stage/cs_score_data.tar.gz" \
    "$data"/rows_student.jsonl "$data"/rows_teacher.jsonl "$data"/canaries.jsonl "$data"/manifest.json)
cp "$repo/experiments/adversarial_confession_strip/vm_score.sh" "$stage/cs_vm_score.sh"
(cd "$stage" && shasum -a 256 cs_gleipnir.bundle cs_score_data.tar.gz cs_vm_score.sh > CS_SHA256SUMS)
echo "staged in $stage at $(git -C "$repo" rev-parse --short HEAD):"
ls -l "$stage"
