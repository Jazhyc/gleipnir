# Infrastructure

## Local cluster

Run `./setup_dev.sh` from a login node, then submit GPU work through Slurm. The
starter template is `cluster/slurm/train_deception_distillation.sh`. It uses one
RTX Pro 6000, one CPU, and 32 GB of RAM and writes the durable log under
`logs/slurm/deception_distillation/`.

Large caches default to `/scratch/$USER`. Override `HF_HOME`,
`HF_HUB_CACHE`, `HF_DATASETS_CACHE`, or `UV_CACHE_DIR` in `.env` when needed.

## Lambda Cloud

The reserved instance title is `monitor-foundation` and exposes two H100 GPUs.
The helper accepts exact console-created titles, so it can address that instance
without renaming or recreating it:

```bash
python scripts/lambda_cloud.py instances --campaign monitor-foundation
python scripts/lambda_cloud.py probe --campaign monitor-foundation
python scripts/lambda_cloud.py sync-commit --campaign monitor-foundation
python scripts/lambda_cloud.py bootstrap --campaign monitor-foundation
python scripts/lambda_cloud.py sync-secrets --campaign monitor-foundation \
  --name HF_TOKEN --name WANDB_API_KEY --name OPENROUTER_API_KEY
python scripts/lambda_cloud.py ssh --campaign monitor-foundation
python scripts/lambda_cloud.py status --campaign monitor-foundation \
  --remote-path results/data_scaling/lambda_status.json
```

The Lambda API key remains local. Selected experiment credentials are sent over
SSH standard input to `~/.config/gleipnir/secrets.env` with mode `600`. The
bootstrap writes cache and CUDA settings to
`~/.config/gleipnir/runtime.env`.

`sync-commit` transfers a committed snapshot. Use `push` for an explicit ignored
input and `pull` for result collection. The helper never terminates an instance
unless `terminate --yes` is invoked; do not do that without explicit user
authorization.

All SSH operations use the same non-interactive, fail-fast transport settings.
Connections are multiplexed through a git-ignored control socket under `.lambda/`
and retained for ten minutes, so repeated commands and transfers reuse an
authenticated session without keeping a two-week connection open. Keepalives
detect an unresponsive session after roughly 45 seconds. Rsync uploads and
downloads retain partial files, enforce a 60-second I/O timeout, and retry up to
three times with bounded backoff. The `status` command accepts only a
project-relative path and refuses to read more than 1 MiB; prefer it to tailing a
large log that is still being written.

## OpenRouter

`gleipnir-openrouter` reads prompt records from JSONL and checkpoints binary
label logprobs to another JSONL file. Each input row needs `id` and `prompt`;
arbitrary other fields are copied into `metadata` when explicitly supplied by
the caller. Example:

```bash
gleipnir-openrouter \
  --input data/teacher_prompts.jsonl \
  --output results/deception_distillation/teacher/openrouter.jsonl \
  --model qwen/qwen3.5-397b-a17b \
  --provider-only Alibaba \
  --concurrency 16
```

The command requires `OPENROUTER_API_KEY`, requests literal terminal `0|1`
logprobs, stores prompt hashes, and resumes only when cached hashes match.
