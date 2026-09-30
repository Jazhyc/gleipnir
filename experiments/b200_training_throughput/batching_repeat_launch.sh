#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced source commit}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="/workspace/gleipnir/.cache/kernels/fa4:${PYTHONPATH:-}"
mkdir -p logs/runpod/b200_training_throughput_batching_repeat
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_training_throughput/batching_repeat_config.yaml \
    > logs/runpod/b200_training_throughput_batching_repeat/preparation.log 2>&1
python3 - <<'PY'
import json
import time
from pathlib import Path

root = Path('/workspace/gleipnir')
source = root / 'results/b200_training_throughput/compile_cache/h100-recipe-b1'
cache = root / '.cache/training/qwen35_4b_b200_fa4'
cache.mkdir(parents=True, exist_ok=True)
destination = cache / 'gpu-0'
assert source.is_dir()
if not destination.exists():
    destination.symlink_to(source, target_is_directory=True)
assert destination.resolve() == source.resolve()
(root / 'results/b200_training_throughput_batching_repeat/runtime_cache_reuse.json').write_text(json.dumps({
    'recorded_at_unix': time.time(), 'source': str(source),
    'destination': str(destination), 'cache_state': 'completed B200 batch-1 and batch-2 caches',
    'overlay': str(root / '.cache/kernels/fa4'),
    'selection_rule': 'complete matched ten-update loops with successful batch-2 longest-row preflight',
}, indent=2) + '\n')
PY
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_training_throughput_batching_repeat/resolved_config.json \
    --revision "$GLEIPNIR_COMMIT" \
    > logs/runpod/b200_training_throughput_batching_repeat/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_training_throughput_batching_repeat/launcher.pid
cat results/b200_training_throughput_batching_repeat/launcher.pid
