#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced source commit}"
export FLA_DISABLE_BACKEND_DISPATCH=1
mkdir -p logs/runpod/b200_training_throughput_warm
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
    --config experiments/b200_training_throughput/warm_config.yaml \
    > logs/runpod/b200_training_throughput_warm/preparation.log 2>&1
python3 - <<'PY'
import json
import time
from pathlib import Path

root = Path('/workspace/gleipnir/results/b200_training_throughput_warm')
old = root.with_name('b200_training_throughput')
assert json.loads((old / 'status.json').read_text())['state'] == 'failed'
cache = root / 'compile_cache'
cache.mkdir(parents=True, exist_ok=True)
sources = {'preflight': old / 'compile_cache' / 'preflight'}
for name in ('h100-recipe-b1', 'half-checkpoint-b1', 'length-grouped-b2'):
    sources[name] = old / 'compile_cache' / 'h100-recipe-b1'
links = []
for name, source in sources.items():
    destination = cache / name
    assert source.is_dir(), source
    assert not destination.exists() and not destination.is_symlink(), destination
    destination.symlink_to(source, target_is_directory=True)
    links.append({'condition': name, 'source': str(source), 'destination': str(destination)})
(root / 'runtime_cache_reuse.json').write_text(json.dumps({
    'recorded_at_unix': time.time(),
    'cache_state': 'reuse_verified_B200_caches_on_identical_software',
    'selection_rule': 'compare complete matched ten-update loops; repeat any promising larger batch after its new graphs are cached',
    'links': links,
}, indent=2) + '\n')
PY
nohup .venv/bin/python -u -m gleipnir.monitoring_systems_screen run \
    --config results/b200_training_throughput_warm/resolved_config.json \
    --revision "$GLEIPNIR_COMMIT" \
    > logs/runpod/b200_training_throughput_warm/launcher.log 2>&1 < /dev/null &
echo "$!" > results/b200_training_throughput_warm/launcher.pid
cat results/b200_training_throughput_warm/launcher.pid
