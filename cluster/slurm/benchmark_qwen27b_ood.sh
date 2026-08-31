#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-qwen27b-ood
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/tool_trajectory_monitoring
log="logs/slurm/tool_trajectory_monitoring/qwen27b-ood-${SLURM_JOB_ID}.out"
exec >"${log}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -m experiments.tool_trajectory_monitoring.benchmark_qwen_ood \
    --config experiments/tool_trajectory_monitoring/qwen27b_ood_benchmark.json \
    --output results/tool_trajectory_monitoring/qwen35_27b_teacher_ood \
    "$@"
