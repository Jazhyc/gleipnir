#!/usr/bin/env bash
# Host-overhead fixes: timing runs (control, trim), then a diagnostic sync audit.
set -euo pipefail
cd /workspace/gleipnir
experiments/b200_training_trim/screen.sh control_host trim_host
GLEIPNIR_HOST_PROFILE_DIR=/workspace/gleipnir/results/b200_training_trim/host_profile_trim_host \
  experiments/b200_training_trim/screen.sh trim_host_audit
echo HOST_OVERHEAD_DONE
