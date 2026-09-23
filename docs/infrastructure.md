# Infrastructure

## Workstation

Without environment modules, `./setup_dev.sh` installs the locked environment
using uv-managed Python 3.12 and project-local ignored caches. Activate with
`source .venv/bin/activate`; set `HF_HOME="$PWD/.cache/huggingface"` in the
inference shell to reuse the local model cache. Existing cache environment
overrides remain supported. See the
[local inference preparation](research/local_inference_throughput.md) for the
RTX 4080 hardware inventory, proposed ID screening set, and runtime budget.

## Local cluster

Run `./setup_dev.sh` from a login node, then submit GPU work through Slurm. The
starter template is `cluster/slurm/train_deception_distillation.sh`. It uses one
RTX Pro 6000, one CPU, and 32 GB of RAM and writes the durable log under
`logs/slurm/deception_distillation/`.

Large caches default to `/scratch/$USER`. Override `HF_HOME`,
`HF_HUB_CACHE`, `HF_DATASETS_CACHE`, or `UV_CACHE_DIR` in `.env` when needed.

## Lambda Cloud

No Lambda Cloud training target is currently reserved. The two-H100
`gleipnir-improvement` instance was terminated on 2026-09-08 following explicit
user authorization and verified artifact collection. See the
[shutdown inventory](findings/gleipnir_improvement_shutdown_inventory.md).
The former `gleipnir-control` and `monitor-foundation` targets are also no longer
active. References to these targets in historical experiment READMEs describe
completed campaigns and are not current launch instructions.

For any separately authorized future target, probe and record its concrete GPU
model and count before freezing a training recipe. The helper accepts exact
console-created titles. These commands document how the former target was
managed:

```bash
python scripts/lambda_cloud.py instances --campaign gleipnir-improvement
python scripts/lambda_cloud.py probe --campaign gleipnir-improvement
python scripts/lambda_cloud.py sync-commit --campaign gleipnir-improvement
python scripts/lambda_cloud.py bootstrap --campaign gleipnir-improvement
python scripts/lambda_cloud.py sync-secrets --campaign gleipnir-improvement \
  --name HF_TOKEN --name WANDB_API_KEY --name OPENROUTER_API_KEY
python scripts/lambda_cloud.py ssh --campaign gleipnir-improvement
```

These historical entries do not authorize launching replacement capacity or
terminating any other instance.

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

The mixed Qwen3.5-4B launcher also source-builds pinned
`causal-conv1d==1.6.2.post1` without dependencies into
`/tmp/gleipnir-qwen35-causal-conv1d`. Its combined preflight executes BF16
forward and backward on the GPU and requires Transformers to bind both the FLA
and causal-convolution fast paths before loading the model. On Stack 24, export
`CUDA_HOME=/usr/local/cuda`: otherwise TileLang can pair the Python environment's
CUDA 13.2 compiler with its CUDA 13.3 CCCL headers and fail JIT compilation. The
shared fast-kernel environment sets this automatically when that toolkit path
exists. The source build is cached by `uv`, but a cold build took about ten
minutes on `gleipnir-improvement`.

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
  --model moonshotai/kimi-k3 \
  --provider-only makora \
  --no-allow-fallbacks \
  --concurrency 16
```

The command requires `OPENROUTER_API_KEY`, requests literal terminal `0|1`
logprobs, stores prompt and request-setting hashes, and resumes only when both
identities match. Kimi K3 is currently available as `moonshotai/kimi-k3`; Makora
is the lowest-priced OpenRouter endpoint with logprob and prompt-cache support.
Keep `--provider-only makora` when endpoint consistency matters. For a more
availability-oriented campaign, use `--provider-order makora --provider-order
fireworks` and leave fallbacks enabled. These are provider tags, not display
names, and are deliberately lowercase.

Provider caching uses an explicit ephemeral `cache_control` breakpoint on the
exact prefix shared by every input prompt. The CLI also hashes that prefix into
a stable OpenRouter `session_id`, completes one request synchronously to warm it,
then fans out at the requested concurrency. OpenRouter can consequently keep the
campaign on the endpoint holding the prefix cache. Use `--session-id` to share a
deliberate cache session across separately materialized shards, or
`--no-explicit-cache`, `--no-sticky-routing`, and `--no-warm-cache` to disable
the corresponding behavior. Keep the long, invariant teacher instructions at
the beginning of every prompt; only the exact shared prefix is reusable.

Each result row records the requested settings and their SHA-256 identity, the
actual model/provider, router metadata, raw usage, timestamps, and normalized
`cache_usage.cached_tokens`, `cache_write_tokens`, and `cache_discount`. The CLI
prints aggregate cache reads/writes and the cached fraction of input tokens at
shutdown. A scale-up should first run a small canary and confirm nonzero cached
tokens after the warm-up row.

On 2026-08-30, a two-request synthetic canary pinned to `makora` confirmed both
`logprobs`/`top_logprobs` and explicit prefix caching. The warm request reported
0 cached tokens; the second request reported 8,448 cached tokens out of 9,627
input tokens. Reported request cost fell from $0.024404 to $0.005218. Current
OpenRouter catalog rates were $2.55/M input, $12.75/M output, and $0.256/M cache
read for Makora, versus $3.00/M, $15.00/M, and $0.30/M for base Fireworks.
