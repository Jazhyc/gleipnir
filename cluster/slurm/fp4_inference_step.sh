#!/usr/bin/env bash
# Execute inside an existing srun GPU allocation; submit no new capacity.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || $# -lt 1 ]]; then
    echo "Expected an existing Slurm allocation and a condition or module" >&2
    exit 2
fi
if [[ "$1" == "--module" ]]; then
    shift
    if [[ $# -lt 1 || ! "$1" =~ ^experiments\.(fp4_inference\.[a-z_]+|local_inference\.profile)$ ]]; then
        echo "Unsupported bounded inference module" >&2
        exit 2
    fi
    fp4_step_module="$1"
    shift
elif [[ $# != 1 ]]; then
    echo "Expected exactly one frozen condition" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
module load CUDA/13.2.0
export HF_HOME="${HF_HOME:-/scratch/${USER}/.huggingface}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -n "${fp4_step_module:-}" ]]; then
    fp4_step_seconds="$(python -c 'from experiments.fp4_inference.run import remaining_seconds, TERMINATION_MARGIN_SECONDS; print(max(0, min(900, int(remaining_seconds() - TERMINATION_MARGIN_SECONDS))))')"
    if (( fp4_step_seconds < 120 )); then
        echo "Insufficient time before the campaign deadline" >&2
        exit 2
    fi
    exec timeout --signal=TERM --kill-after=15s "$fp4_step_seconds" \
        python -u -m "$fp4_step_module" "$@"
fi
exec python -u -m experiments.fp4_inference.run --condition "$1"
