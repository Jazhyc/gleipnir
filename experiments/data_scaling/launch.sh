#!/usr/bin/env bash
# Launch the prepared local Slurm sweep and its dependent evaluation job.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

student_rows="${KIMI_STUDENT_ROWS:-/scratch/s4626451/Aletheias-Quest-Competition/results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/train/student_rows.jsonl}"
soft_targets="${KIMI_SOFT_TARGETS:-/scratch/s4626451/Aletheias-Quest-Competition/results/blackbox/kimi_k3_tvg_soft_full_plus_liars_v1/train/soft_targets.jsonl}"
validation_split="${COMPETITION_VALIDATION_SPLIT:-/scratch/s4626451/Aletheias-Quest-Competition/dev_splits/dry.validation.yaml}"

source .venv/bin/activate
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
python experiments/data_scaling/prepare.py \
  --student-rows "${student_rows}" \
  --soft-targets "${soft_targets}" \
  --validation-split "${validation_split}"

job_count="$(wc -l < results/data_scaling/jobs.jsonl)"
if [[ "${job_count}" -lt 1 ]]; then
  echo "prepared no scaling jobs" >&2
  exit 1
fi
last_index="$((job_count - 1))"
train_job="$(
  sbatch --parsable --array="0-${last_index}" \
    cluster/slurm/train_data_scaling_array.sh
)"
eval_job="$(
  sbatch --parsable --dependency="afterok:${train_job}" \
    cluster/slurm/evaluate_data_scaling.sh
)"
echo "submitted training array ${train_job} and dependent evaluation ${eval_job}"
