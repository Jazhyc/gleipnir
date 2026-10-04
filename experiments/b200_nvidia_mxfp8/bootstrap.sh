#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
overlay="$PWD/.cache/kernels/nvidia_mxfp8"
artifact="$PWD/results/b200_nvidia_mxfp8"
mkdir -p "$overlay" "$artifact/setup" logs/runpod/b200_nvidia_mxfp8
sha256sum "$artifact/setup/cudnn-source.tar.gz" > "$artifact/setup/source.sha256"
mkdir -p "$overlay/source"
tar --no-same-owner -xzf "$artifact/setup/cudnn-source.tar.gz" -C "$overlay/source"
uv pip install --python .venv/bin/python --target "$overlay/runtime" --no-deps nvidia-cudnn-cu13==9.26.0.51
uv pip install --python .venv/bin/python --target "$overlay/build" --no-deps cmake==4.4.4 ninja==1.13.2 'pybind11[global]==2.13.6' setuptools==84.0.0 wheel==0.48.0
export CUDNN_PATH="$overlay/runtime/nvidia/cudnn"
export CUDAToolkit_ROOT=/usr/local/cuda
export PATH="$overlay/build/bin:$PATH"
export PYTHONPATH="$overlay/build:$PYTHONPATH"
export CMAKE_PREFIX_PATH="$overlay/build/pybind11/share/cmake/pybind11"
export CMAKE_BUILD_PARALLEL_LEVEL=4
uv pip install --python .venv/bin/python --target "$overlay/frontend" --no-deps --no-build-isolation "$overlay/source"
printf '%s\n' 'setup_complete=true'
