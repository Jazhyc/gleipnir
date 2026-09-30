#!/usr/bin/env bash
# Keep the locked environment and caches on the Pod's persistent workspace.
set -euo pipefail
cd /workspace/gleipnir
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$PWD/.cache/flashinfer"
export PYTHONPATH="$PWD/src:$PWD"
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=4
mkdir -p logs/runpod/runpod_gleipnir4b_id results/runpod_gleipnir4b_id
cat > .cache-runtime.env <<'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH="/workspace/gleipnir/.venv/bin:$CUDA_HOME/bin:$PATH"
export HF_HOME=/workspace/gleipnir/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR=/workspace/gleipnir/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/gleipnir/.cache/python
export TRITON_CACHE_DIR=/workspace/gleipnir/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/gleipnir/.cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/workspace/gleipnir/.cache/flashinfer
export PYTHONPATH=/workspace/gleipnir/src:/workspace/gleipnir
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=4
EOF
uv sync --python 3.12 --locked
source .cache-runtime.env
.venv/bin/python - <<'PY'
import json
import platform
import subprocess
from pathlib import Path
import torch
import transformers
import vllm

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Exactly one visible CUDA GPU is required")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
torch.cuda.synchronize()
record = {
    "python": platform.python_version(), "torch": torch.__version__,
    "transformers": transformers.__version__, "vllm": vllm.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
    "uv_lock_sha256": __import__("hashlib").sha256(Path("uv.lock").read_bytes()).hexdigest(),
}
Path("results/runpod_gleipnir4b_id/hardware.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
PY
