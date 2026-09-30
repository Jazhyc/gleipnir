#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced source commit}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="/workspace/gleipnir/.cache/kernels/fa4:${PYTHONPATH:-}"
mkdir -p logs/runpod/b200_training_throughput_fa4
# GPU probes are serialized after the preceding screen completes.
.venv/bin/python experiments/b200_training_throughput/fa4_kernel_canary.py \
    > logs/runpod/b200_training_throughput_fa4/kernel_canary.log 2>&1
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_training_throughput/fa4_config.yaml \
    > logs/runpod/b200_training_throughput_fa4/preparation.log 2>&1
python3 - <<'PY'
import json
import time
from pathlib import Path

root = Path('/workspace/gleipnir')
old = root / 'results/b200_training_throughput/compile_cache/h100-recipe-b1'
cache = root / '.cache/training/qwen35_4b_b200_fa4'
cache.mkdir(parents=True, exist_ok=True)
destination = cache / 'gpu-0'
assert old.is_dir()
assert not destination.exists() and not destination.is_symlink()
destination.symlink_to(old, target_is_directory=True)
result = root / 'results/b200_training_throughput_fa4'
(result / 'runtime_cache_reuse.json').write_text(json.dumps({
    'recorded_at_unix': time.time(), 'source': str(old),
    'destination': str(destination), 'cache_state': 'existing SDPA/FLA caches; FA4 initially cold',
    'overlay': str(root / '.cache/kernels/fa4'),
    'selection_rule': 'complete matched loops; warmed repeat of FA4 if initially promising',
}, indent=2) + '\n')
PY
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_training_throughput_fa4/resolved_config.json \
    --revision "$GLEIPNIR_COMMIT" \
    > logs/runpod/b200_training_throughput_fa4/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_training_throughput_fa4/launcher.pid
cat results/b200_training_throughput_fa4/launcher.pid
