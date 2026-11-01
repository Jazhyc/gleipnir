#!/usr/bin/env bash
# Local (Mac): stage everything the VM needs for the lean main run (Amendment 2)
# in one folder, for `gcloud compute scp`. Run from anywhere after committing:
#
#   experiments/adversarial_monitor_aware/pack.sh [STAGE_DIR]
#
# Stages:
#   gleipnir.bundle   git bundle of feat/adversarial-stress-tests (HEAD)
#   ma_data.tar.gz    ignored inputs, paths relative to the repository root:
#                       data/adversarial_cheating_pilot/impossiblebench/
#                         (ImpossibleBench checkout at d4f9a14, with its .git)
#                       results/adversarial_cheating_pilot/t_b_conflicting/*.eval
#                       results/adversarial_cheating_pilot/t_b_conflicting/recheck.jsonl
#                         (the pilot runs reused as `natural`)
#   vm_setup.sh       this study's setup script (also inside the bundle)
#   SHA256SUMS
# The Tinker key is not staged; it goes to the VM separately as ~/tinker.env.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
stage=${1:-$repo/../../cache/ma_vm_stage}
branch=feat/adversarial-stress-tests
nat=results/adversarial_cheating_pilot/t_b_conflicting

[ "$(git -C "$repo" branch --show-current)" = "$branch" ] || { echo "not on $branch"; exit 1; }
git -C "$repo" diff --quiet HEAD -- experiments/adversarial_monitor_aware \
    || { echo "uncommitted study changes; commit first"; exit 1; }
[ "$(git -C "$repo/data/adversarial_cheating_pilot/impossiblebench" rev-parse HEAD)" \
    = d4f9a149db2b9cec0fed7bab01d91c07ee10aca2 ] || { echo "ImpossibleBench not at d4f9a14"; exit 1; }
[ "$(ls "$repo/$nat"/*.eval | wc -l | tr -d ' ')" = 1 ] || { echo "expected one pilot .eval"; exit 1; }

mkdir -p "$stage"
rm -f "$stage"/gleipnir.bundle "$stage"/ma_data.tar.gz "$stage"/SHA256SUMS
git -C "$repo" bundle create "$stage/gleipnir.bundle" "$branch"
(cd "$repo" && COPYFILE_DISABLE=1 tar czf "$stage/ma_data.tar.gz" --exclude=__pycache__ \
    data/adversarial_cheating_pilot/impossiblebench \
    "$nat"/*.eval "$nat"/recheck.jsonl)
cp "$repo/experiments/adversarial_monitor_aware/vm_setup.sh" "$stage/"
(cd "$stage" && shasum -a 256 gleipnir.bundle ma_data.tar.gz vm_setup.sh > SHA256SUMS)
echo "staged in $stage at $(git -C "$repo" rev-parse --short HEAD):"
ls -l "$stage"
