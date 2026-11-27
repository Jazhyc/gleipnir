# Infrastructure

Platform pages:

- [Lambda Cloud](lambda_cloud.md)
- [Runpod](runpod.md), with dated machine events in [Runpod history](runpod_history.md)
- [OpenRouter](openrouter.md)

## Workstation

Without environment modules, `./setup_dev.sh` installs the locked environment
using uv-managed Python 3.12 and project-local ignored caches. Activate with
`source .venv/bin/activate`; set `HF_HOME="$PWD/.cache/huggingface"` in the
inference shell to reuse the local model cache. Existing cache environment
overrides remain supported. See the
[local inference preparation](../plans/local_inference_throughput.md) for the
RTX 4080 hardware inventory, proposed ID screening set, and runtime budget.

## Local cluster

Run `./setup_dev.sh` from a login node, then submit GPU work through Slurm. The
starter template is `cluster/slurm/train_deception_distillation.sh`. It uses one
RTX Pro 6000, one CPU, and 32 GB of RAM and writes the durable log under
`logs/slurm/deception_distillation/`.

Large caches default to `/scratch/$USER`. Override `HF_HOME`,
`HF_HUB_CACHE`, `HF_DATASETS_CACHE`, or `UV_CACHE_DIR` in `.env` when needed.
