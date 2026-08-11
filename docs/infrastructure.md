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

Qwen3.5 training launchers install the two pinned FLA 0.5.2 packages without
dependencies into `/tmp/gleipnir-qwen35-fla`, prepend that isolated directory to
`PYTHONPATH`, and run an import/kernel probe before starting an expensive job.
This deliberately leaves the locked project environment unchanged while making
the accelerated gated-delta implementation reproducible after an ephemeral
instance reboot. The adapter-capacity campaign additionally locks bitsandbytes
and uses standard NF4 QLoRA so microbatch 8 remains viable at high ranks.

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

The SSH transport runs through `scripts/tcp_mss_proxy.py`, which advertises a
conservative 1400-byte TCP maximum segment size. This avoids payload-dependent
stalls observed between the Habrok compute network and Lambda while retaining
normal SSH authentication and encryption. In a live transfer probe, ordinary
SCP stalled without transferring data, whereas MSS-capped rsync transferred a
64 MiB incompressible file, resumed correctly after a forced interruption, and
matched its remote SHA-256. The same path transferred a 116 MiB LoRA adapter in
15 seconds with a matching checksum. The proxy uses only the Python standard
library and is applied automatically to SSH, rsync, push, pull, and code sync.

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
