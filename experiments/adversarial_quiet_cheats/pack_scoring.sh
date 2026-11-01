#!/usr/bin/env bash
# Local (Mac): stage what the A100 VM needs to score arm 1, after the rows
# freeze commit (config arm1_deletion.rows_sha256 set):
#
#   experiments/adversarial_quiet_cheats/pack_scoring.sh [STAGE_DIR]
#
# Stages gleipnir.bundle (feat/adversarial-stress-tests), qc_score_data.tar.gz
# (arm 1 rows and manifest, plus monitor-aware's canaries), vm_score.sh and
# SHA256SUMS. No secrets; GPU scoring needs none.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
stage=${1:-$repo/../../cache/qc_score_stage}
branch=feat/adversarial-stress-tests
data=data/adversarial_quiet_cheats/arm1

[ "$(git -C "$repo" branch --show-current)" = "$branch" ] || { echo "not on $branch"; exit 1; }
git -C "$repo" diff --quiet HEAD -- experiments/adversarial_quiet_cheats \
    || { echo "uncommitted study changes; commit first"; exit 1; }
(cd "$repo" && python3 - <<'PYEOF'
import hashlib, json
from pathlib import Path
arm1 = json.load(open("experiments/adversarial_quiet_cheats/config.json"))["arm1_deletion"]
ma = json.load(open("experiments/adversarial_monitor_aware/config.json"))["scoring"]
frozen = arm1.get("rows_sha256") or {}
for key in ("rows_student", "rows_teacher"):
    assert frozen.get(key), f"{key}: rows freeze not done"
    path = Path("data/adversarial_quiet_cheats/arm1") / f"{key}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[key], path
path = Path(ma["data"]) / "canaries.jsonl"
assert hashlib.sha256(path.read_bytes()).hexdigest() == ma["sha256"]["canaries"], path
print("inputs match their frozen SHA256")
PYEOF
)
mkdir -p "$stage"
rm -f "$stage"/gleipnir.bundle "$stage"/qc_score_data.tar.gz "$stage"/SHA256SUMS
git -C "$repo" bundle create "$stage/gleipnir.bundle" "$branch"
(cd "$repo" && COPYFILE_DISABLE=1 tar czf "$stage/qc_score_data.tar.gz" \
    "$data"/rows_student.jsonl "$data"/rows_teacher.jsonl "$data"/manifest.json \
    data/adversarial_monitor_aware/canaries.jsonl)
cp "$repo/experiments/adversarial_quiet_cheats/vm_score.sh" "$stage/"
(cd "$stage" && shasum -a 256 gleipnir.bundle qc_score_data.tar.gz vm_score.sh > SHA256SUMS)
echo "staged in $stage at $(git -C "$repo" rev-parse --short HEAD):"
ls -l "$stage"
