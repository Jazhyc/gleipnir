#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
target=/workspace/gleipnir/.cache/kernels/fa4
mkdir -p "$target" logs/runpod/b200_training_throughput_fa4
cp experiments/b200_training_throughput/fa4_sitecustomize.py "$target/sitecustomize.py"
uv pip install --python .venv/bin/python --target "$target" --no-deps \
    -r experiments/b200_training_throughput/fa4_requirements.txt
# CUDA-specific wheels overlap the default CuTe namespace. Restore CUDA 13 last.
uv pip install --python .venv/bin/python --target "$target" --no-deps \
    --reinstall-package nvidia-cutlass-dsl-libs-cu13 \
    nvidia-cutlass-dsl-libs-cu13==4.8.0
PYTHONPATH="$target:${PYTHONPATH:-}" .venv/bin/python - <<'PY'
import base64
import hashlib
import importlib.metadata as m
import json
from pathlib import Path

target = Path('/workspace/gleipnir/.cache/kernels/fa4')
packages = {}
for line in Path('experiments/b200_training_throughput/fa4_requirements.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, expected = line.split('==')
    actual = m.version(name)
    assert actual == expected, (name, actual, expected)
    packages[name] = actual
dist = m.distribution('nvidia-cutlass-dsl-libs-cu13')
checked = 0
for item in dist.files or []:
    if not item.hash or item.hash.mode != 'sha256':
        continue
    digest = base64.urlsafe_b64encode(hashlib.sha256(dist.locate_file(item).read_bytes()).digest()).decode().rstrip('=')
    assert digest == item.hash.value, str(item)
    checked += 1
assert checked > 0
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
import cutlass
assert Path(cutlass.__file__).is_relative_to(target), cutlass.__file__
Path('logs/runpod/b200_training_throughput_fa4/dependencies.json').write_text(json.dumps({
    'overlay': str(target), 'packages': packages,
    'cuda13_record_files_verified': checked,
    'torch': m.version('torch'), 'transformers': m.version('transformers'),
    'flash_attn_func_module': flash_attn_func.__module__,
    'flash_attn_varlen_func_module': flash_attn_varlen_func.__module__,
    'cutlass_module': cutlass.__file__,
}, indent=2) + '\n')
print('FA4 isolated import and CUDA 13 wheel integrity passed', packages)
PY
