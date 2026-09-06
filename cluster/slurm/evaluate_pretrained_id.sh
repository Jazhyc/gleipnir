#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-pretrained-id
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=alma9
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
size="${1:?Specify 4b or 9b}"
[[ "$size" == 4b || "$size" == 9b ]]
mkdir -p logs/slurm/pretrained_id
exec >"logs/slurm/pretrained_id/${size}-${SLURM_JOB_ID}.out" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"
module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
source .venv/bin/activate
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
export HF_HOME="${HF_HOME:-/scratch/${USER}/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python -m experiments.pretrained_id.run --size "$size"
