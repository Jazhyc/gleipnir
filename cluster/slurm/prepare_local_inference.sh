#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-inference-prep
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm/local_inference
exec >"logs/slurm/local_inference/prepare-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/scratch/${USER}/.huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
if [[ "${1:-}" == "--reference" ]]; then
    nvidia-smi
fi
python -m experiments.local_inference.prepare
python -m experiments.local_inference.prepare32
printf '%s\n' \
    'f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf  data/local_inference/subset.jsonl' \
    'aadb48556b134150e43946ba39d31512498e2d61b8a88a7a24fd9617b5633f03  data/local_inference/iteration32.jsonl' \
    | sha256sum --check
python -m experiments.local_inference.merge
if [[ "${1:-}" == "--reference" ]]; then
    python -m experiments.local_inference.reference
fi
printf 'Requested preparation stages completed.\n'
