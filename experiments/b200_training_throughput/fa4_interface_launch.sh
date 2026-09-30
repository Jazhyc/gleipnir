#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced source commit}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="/workspace/gleipnir/.cache/kernels/fa4:${PYTHONPATH:-}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/gleipnir/.cache/training/fa4_4.0.0b33_cute
mkdir -p logs/runpod/b200_training_throughput_fa4_interface
# The initial FA4 kernel probe already passed; preserve its report and failed preflight.
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_training_throughput/fa4_interface_config.yaml \
    > logs/runpod/b200_training_throughput_fa4_interface/preparation.log 2>&1
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
if not destination.exists():
    destination.symlink_to(old, target_is_directory=True)
assert destination.resolve() == old.resolve()
result = root / 'results/b200_training_throughput_fa4_interface'
(result / 'runtime_cache_reuse.json').write_text(json.dumps({
    'recorded_at_unix': time.time(), 'source': str(old),
    'destination': str(destination), 'cache_state': 'completed FA4 kernel and longest-row compilation; same shared cache for both backends',
    'overlay': str(root / '.cache/kernels/fa4'),
    'fa4_persistent_cache_enabled': True,
    'fa4_persistent_cache_dir': str(root / '.cache/training/fa4_4.0.0b33_cute'),
    'selection_rule': 'complete matched loops; warmed repeat of FA4 if initially promising',
}, indent=2) + '\n')
PY
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_training_throughput_fa4_interface/resolved_config.json \
    --revision "$GLEIPNIR_COMMIT" \
    > logs/runpod/b200_training_throughput_fa4_interface/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_training_throughput_fa4_interface/launcher.pid
cat results/b200_training_throughput_fa4_interface/launcher.pid
