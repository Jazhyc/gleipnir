#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-classifier-throughput
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/slurm/classifier_throughput
log="logs/slurm/classifier_throughput/benchmark-${SLURM_JOB_ID}.out"
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
export TOKENIZERS_PARALLELISM=false

python experiments/classifier_throughput/benchmark.py \
    --output-dir "results/classifier_throughput/${SLURM_JOB_ID}"
