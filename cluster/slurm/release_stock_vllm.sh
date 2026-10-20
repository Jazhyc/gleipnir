#!/usr/bin/env bash
#SBATCH --job-name=gleipnir-stock-vllm
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/%x-%j.bootstrap.out
# Validate the documented stock `vllm serve` recipe for a released BF16
# checkpoint against the recorded BF16 rows. 8 CPUs/64 GB: vLLM's API server,
# tokenizer and engine processes run beside the scoring client. The download
# goes to a job-scoped cache that is deleted at exit (cluster storage is tight).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL=${1:-4b}
mkdir -p logs/slurm/release_serving results/release_serving/stock_vllm
log="logs/slurm/release_serving/stock-vllm-${MODEL}-${SLURM_JOB_ID}.out"
exec >"${log}" 2>&1
rm -f "logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.bootstrap.out"

module load Python/3.12.3-GCCcore-13.3.0 CUDA/13.2.0
# The locked vLLM 0.31 environment (the default `.venv` may hold older pins).
source "${VENV:-.venv-vllm031}/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
export PYTHONPATH=src:.
export HF_HOME="/scratch/${USER}/.cache-stock-vllm-${SLURM_JOB_ID}"
trap 'kill ${SERVER:-} 2>/dev/null || true; rm -rf "${HF_HOME}"' EXIT
nvidia-smi -L

mapfile -t CMD < <(python -c "
from experiments.release_serving.stock_vllm import serve_command
from experiments.release_serving.targets import TARGETS
t = TARGETS['${MODEL}']
print('\n'.join(serve_command(t.merged_repo, t.version)))")
printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}" > "logs/slurm/release_serving/stock-vllm-server-${MODEL}-${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
started=$(date +%s)
until curl -sf localhost:8000/health >/dev/null; do
  kill -0 ${SERVER} 2>/dev/null || { echo SERVER_EXITED; tail -40 "logs/slurm/release_serving/stock-vllm-server-${MODEL}-${SLURM_JOB_ID}.log"; exit 1; }
  sleep 5
done
echo "ready after $(( $(date +%s) - started )) s"
python -m experiments.release_serving.stock_vllm "${MODEL}" \
  --output "results/release_serving/stock_vllm/${MODEL}.json"
echo STOCK_VLLM_DONE
