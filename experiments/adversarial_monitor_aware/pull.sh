#!/usr/bin/env bash
# Local (Mac): copy the lean main run's results and logs back from the VM into
#   results/adversarial_monitor_aware/main/
#   logs/gcp/adversarial_monitor_aware/   (with ma_status.log and vm_run.out)
# Safe to rerun; existing local files are overwritten by the VM's copies.
#
#   experiments/adversarial_monitor_aware/pull.sh [VM] [ZONE]
set -euo pipefail
vm=${1:-ib-cpu-1}
zone=${2:-asia-south1-a}
project=${GCP_PROJECT:-avid-renderer-503311-n6}
export CLOUDSDK_ACTIVE_CONFIG_NAME=${CLOUDSDK_ACTIVE_CONFIG_NAME:-swarm}
repo=$(cd "$(dirname "$0")/../.." && pwd)
remote=lab/upstream/gleipnir
scp_() { gcloud compute scp --zone="$zone" --project="$project" "$@"; }

mkdir -p "$repo/results/adversarial_monitor_aware" "$repo/logs/gcp"
scp_ --recurse "$vm:$remote/results/adversarial_monitor_aware/main" \
    "$repo/results/adversarial_monitor_aware/"
scp_ --recurse "$vm:$remote/logs/gcp/adversarial_monitor_aware" "$repo/logs/gcp/"
scp_ "$vm:ma_status.log" "$vm:vm_run.out" "$repo/logs/gcp/adversarial_monitor_aware/" \
    || echo "(no vm_run.out yet)"
ls -l "$repo/results/adversarial_monitor_aware/main" "$repo/logs/gcp/adversarial_monitor_aware"
tail -3 "$repo/logs/gcp/adversarial_monitor_aware/ma_status.log"
