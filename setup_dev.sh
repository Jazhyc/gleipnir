#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v module >/dev/null 2>&1; then
  module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
  cache_root="/scratch/${USER}"
  default_hf_home="${cache_root}/.huggingface"
  python_spec="$(command -v python)"
else
  cache_root="$PWD"
  default_hf_home="${cache_root}/.cache/huggingface"
  python_spec="3.12"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/.cache/python}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${GLEIPNIR_VENV:-.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/.cache/uv}"
export HF_HOME="${HF_HOME:-${default_hf_home}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

uv sync --python "$python_spec" --locked

echo "Gleipnir environment ready at ${UV_PROJECT_ENVIRONMENT}."
echo "Activate with: source ${UV_PROJECT_ENVIRONMENT}/bin/activate"
