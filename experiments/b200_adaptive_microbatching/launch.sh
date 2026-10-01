#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced source commit}"
export FLA_DISABLE_BACKEND_DISPATCH=1
mkdir -p logs/runpod/b200_adaptive_microbatching
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_adaptive_microbatching/config.yaml \
    > logs/runpod/b200_adaptive_microbatching/preparation.log 2>&1
python3 - <<'PY'
import json
import time
from pathlib import Path

root = Path('/workspace/gleipnir')
cache = root / '.cache/training/qwen35_4b_b200_fa4/gpu-0'
assert cache.is_dir()
(root / 'results/b200_adaptive_microbatching/runtime_cache_reuse.json').write_text(
    json.dumps({'recorded_at_unix': time.time(), 'cache': str(cache.resolve()),
                'state': 'completed B200 singleton, grouped-2, and FA4 screens',
                'selection_rule': 'profile both budgets; cached unsynchronized repeat before selection'}, indent=2) + '\n')
PY
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_adaptive_microbatching/resolved_config.json \
    --revision "$GLEIPNIR_COMMIT" \
    > logs/runpod/b200_adaptive_microbatching/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_adaptive_microbatching/launcher.pid
cat results/b200_adaptive_microbatching/launcher.pid
