#!/usr/bin/env bash
# Rebuild isolated pinned kernels on a fresh Blackwell workspace.
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
export MAX_JOBS=4
CUDA_ARCHS=$(.venv/bin/python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}{minor}")')
export CUDA_ARCHS
if [[ "$CUDA_ARCHS" != 100 && "$CUDA_ARCHS" != 103 ]]; then
  echo "Unsupported native FP4 pilot GPU architecture: $CUDA_ARCHS" >&2
  exit 1
fi
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS:0:2}.${CUDA_ARCHS:2:1}"
export FORCE_BUILD=1
export FLA_DISABLE_BACKEND_DISPATCH=1
mkdir -p .cache/kernels/{fla,causal_conv1d,triton,fouroversix-1.0.5,sources}
ln -sfn "$PWD/.cache/kernels/fla" /tmp/gleipnir-qwen35-fla
ln -sfn "$PWD/.cache/kernels/causal_conv1d" /tmp/gleipnir-qwen35-causal-conv1d
ln -sfn "$PWD/.cache/kernels/triton" /tmp/gleipnir-triton-3.7.1
.venv/bin/python - <<'PY'
from pathlib import Path
from gleipnir.qwen35_fast_training import ensure_qwen35_long_trajectory_kernels
ensure_qwen35_long_trajectory_kernels(python=Path('.venv/bin/python'))
PY
.venv/bin/python - <<'PY'
import hashlib
import json
import urllib.request
from pathlib import Path
from gleipnir.fouroversix_training import FOUROVERSIX_SDIST_SHA256
path = Path('.cache/kernels/sources/fouroversix-1.0.5.tar.gz')
if not path.exists():
    with urllib.request.urlopen('https://pypi.org/pypi/fouroversix/1.0.5/json') as response:
        metadata = json.load(response)
    artifact = next(item for item in metadata['urls'] if item['packagetype'] == 'sdist')
    urllib.request.urlretrieve(artifact['url'], path)
if hashlib.sha256(path.read_bytes()).hexdigest() != FOUROVERSIX_SDIST_SHA256:
    raise ValueError('Four Over Six source checksum mismatch')
PY
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  --target .cache/kernels/fouroversix-1.0.5 \
  .cache/kernels/sources/fouroversix-1.0.5.tar.gz
export PYTHONPATH="$PWD/.cache/kernels/fouroversix-1.0.5:/tmp/gleipnir-triton-3.7.1:$PYTHONPATH"
.venv/bin/python experiments/b200_fouroversix/kernel_canary.py \
  --output results/fp4_setup/native_kernel_canary.json
.venv/bin/python experiments/b200_fouroversix/kernel_canary.py \
  --backward-mode dequantized_bf16 \
  --output results/fp4_setup/dequantized_kernel_canary.json
