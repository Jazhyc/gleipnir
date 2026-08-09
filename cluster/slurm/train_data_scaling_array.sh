#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-scale-train
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-23
#SBATCH --output=logs/slurm/%x-%A_%a.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/data_scaling
log="logs/slurm/data_scaling/train-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"
exec >"${log}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/${USER}}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python experiments/data_scaling/run_train.py \
  --jobs results/data_scaling/jobs.jsonl \
  --index "${SLURM_ARRAY_TASK_ID}"
