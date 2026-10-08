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

`sync-commit` transfers a committed snapshot. Use `push` for explicit ignored
inputs and `pull` for result collection. Supply several paths after `--local-path`
(push) or `--remote-path` (pull), or repeat that option, to transfer them in one
rsync session, preserving their repository-relative locations. The opposite path
option can rename a single source; batch transfers use matching paths on both
sides. Directories in a batch are copied recursively:

```bash
python scripts/lambda_cloud.py push --campaign example \
  --local-path data/example/input.jsonl data/example/manifest.json
python scripts/lambda_cloud.py pull --campaign example \
  --remote-path results/example/summary.json logs/lambda/example/
```

The helper never terminates an instance unless `terminate --yes` is invoked;
do not do that without explicit user authorization.

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

## Runpod

On 2026-10-01 the user authorized resuming FP4 training work, then explicitly
selected B300 after B200 capacity exhausted. New Pod `mqj2vv2h99ldie`
(`gleipnir-b300-fp4`) was allocated in US-WA-2 at $7.89/hour with a 100 GB Pod
volume mounted at `/workspace` and a 30 GB container disk. That region does not
support network volumes. The Pod volume survives stopping but is deleted on
termination; its listed storage cost is $10/month running or $20/month stopped,
plus $3/month container storage while running. The former B200 network volume
`ixbh81vf9c` remains intact in US-NC-2 and the former Pod remains stopped.

The new container was still initializing at the initial checks, with no runtime
or direct SSH mapping. Do not treat allocation as verified hardware or a started
experiment. Use `--pod-file .runpod/b300.json` with `scripts/runpod_cloud.py`
after saving sanitized live SSH metadata. The B300 experiment contract is
[`fp4_stability`](../experiments/fp4_stability/README.md). Restore local
frozen inputs and rebuild isolated pinned kernels; no cross-region volume
migration has occurred. Preserve historical B200 cache/results identities.

The user subsequently reserved B200 `o87sut99lu3ljs` in US-NC-2, on the same
physical host `hgmwgcbuiv4y` as original Pod `alzfug70g5237b`. The reserved Pod
had no volume, and Runpod forbids adding a network mount after Pod creation.
After stopping the B300 and the empty reservation Pod, the original B200 Pod
successfully resumed with its existing network volume on 2026-10-01. Both unused
Pods are stopped, not terminated. No data migration is required. Verify live
runtime/SSH mapping and the persistent environment before launching diagnostics.

The user authorized one B200 at at most $8/hour on 2026-09-30, with no region
restriction, a B300 fallback only after notification that B200 is unavailable,
and keeping the Pod running after setup and ID inference. Pod `alzfug70g5237b`
(`gleipnir-b200`) was provisioned in `US-NC-2` at $6.79/hour with 100 GB standard
network volume `ixbh81vf9c` ($7/month) mounted at `/workspace` and a 50 GB
container disk ($5/month while running). This is independent of the historical
Lambda targets. Network volumes can grow but cannot shrink.

The official Runpod MCP manages the Pod lifecycle. `scripts/runpod_cloud.py`
handles real SSH and rsync from a sanitized live Pod snapshot in ignored
`.runpod/pod.json`. The external SSH port can change after a restart; refresh
the snapshot from `get-pod` before reconnecting. Keep `RUNPOD_API_KEY` local in
the ignored `.env`; no account API key is copied to the Pod. Code sync excludes
credentials and ignored model/data artifacts, which are transferred explicitly.
Runpod network storage rejects chown, so rsync uses `--no-owner --no-group`.

`push` and `pull` accept multiple repository-relative paths in one rsync session,
preserving their directory structure. Quote names containing spaces. Pulls that
include a directory path ending in `/` omit `*.tmp` files:

```bash
python scripts/runpod_cloud.py push \
  data/example/input.jsonl data/example/manifest.json
python scripts/runpod_cloud.py pull \
  results/example/summary.json logs/runpod/example/
```

The official CUDA-13 image from template `a9dk3g7cny` is
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`. Bootstrap with
`scripts/bootstrap_runpod.sh`: install the checked-in lock using Python 3.12,
keep the virtualenv and caches under `/workspace/gleipnir`, and verify a BF16
GPU matmul before model loading. Bootstrap recorded a B200 (183,359 MiB),
driver 580.126.09, CUDA 13.0, PyTorch 2.11.0+cu130, Transformers 5.14.1, and
vLLM 0.24.0. The first campaign is
[`runpod_gleipnir4b_id`](../experiments/runpod_gleipnir4b_id/README.md): kernel
canary, causal-master/serving parity, then the canonical CoT-removed ID set.
It does not establish a B200 training-throughput recipe.

The cold source build of pinned causal-conv1d took 12m43s; it is needed for
the bounded Transformers master-parity path, while full inference uses vLLM.
The kernel canary passed BF16 forward/backward on B200 with FLA 0.5.2 and
causal-conv1d 1.6.2.post1. The initial locked install exposed the upstream
[CUTLASS shared-file conflict](https://github.com/NVIDIA/cutlass/issues/3170):
vLLM detected divergent CUDA-13 wheel files and selected Triton/FLA GDN prefill.
That attempt was stopped before full evaluation. Bootstrap now reinstalls the
locked CUDA-13 CUTLASS wheel last and verifies its RECORD hashes. The runner
requires explicit activation of FlashInfer GDN prefill in the serving log.
vLLM compilation caches also live in the persistent workspace.

The first full ID evaluation completed: 3,012 finite predictions, passing
master/serving parity, and macro AUROC 0.953021 / pAUROC@20 0.851445.
The Pod was verified RUNNING after artifact collection. See the
[B200 finding](findings/runpod_b200_gleipnir4b_id.md) for numerical agreement,
calibration, measured inference throughput, and the cold-start limitations.

On 2026-10-01, after the B200 training/kernel diagnostics and artifact collection,
the user explicitly requested shutting down compute. Pod `alzfug70g5237b` was
stopped and read back as **EXITED**. Network volume `ixbh81vf9c` remains present;
weights, environments, kernel/compiler caches, datasets and artifacts were
verified on `/workspace` before stopping. Small container-local caches and the
Python runtime were archived there and checksummed locally. GPU compute billing
ended; retained storage still bills. Before a later authorized start, read the
[shutdown and restart record](findings/b200_fouroversix_training.md#subsequent-b200-shutdown-and-restart-state),
refresh live SSH metadata and restore the temporary kernel links. No Pod or
network volume was deleted.

### B200 stopped after the completed campaigns, 2026-10-03

The user explicitly requested deactivating the B200 after finishing the
experiments. Pod `alzfug70g5237b` (`gleipnir-b200`) was stopped and a separate
live read confirmed **EXITED** at 02:58 UTC. A final account inventory showed
both this Pod and reservation Pod `o87sut99lu3ljs` EXITED, with no running Pod.
The independent network-volume read confirmed `ixbh81vf9c`
(`gleipnir-b200-workspace`, US-NC-2, STANDARD, now 200 GB) remains present.
GPU compute billing ends on stop; retained storage continues billing.

Before stopping, direct SSH verified zero GPU processes, 0 MiB allocated,
0% utilization and 30 C, and no training/evaluation worker in the process
inventory. `findmnt` confirmed `/workspace` is the network-volume mount.
Completed student-awareness and honest-code-control artifacts and passing
final audits are available locally; their final audit files were created
locally after collection and need not exist on the Pod. Results and logs
remain on the persistent workspace as well.

Container-local profiler, TorchInductor and FlashInfer caches were archived
to `.cache/runtime-resume/container-compiler-caches-20261003.tar.gz` on the
workspace and collected to the same local path. Both copies have SHA256
`eb7c45fda921ffb0eaf5c970b940f17574f90fc1e6b2dd7a4ea7dd0a6df63ca5`.
The previous Python/runtime archive remains intact with its recorded
`e80c0271...` checksum; the current Python version was 3.12.3. Reproducible
container-local Hugging Face/uv download caches were not included in the new
compiler-cache archive. Existing workspace weights and environments persist.

Sanitized read-backs, volume identity, local audit hashes and preservation
checks are in `results/b200_shutdown/20261003/stop_receipt.json`.
`.runpod/pod.json` records EXITED with runtime and direct SSH mapping cleared;
refresh it from a live read after any separately authorized future start.
Restore the recorded temporary kernel links and container caches if needed.
No Pod or network volume was terminated or deleted.

### B200 restarted for the training-injection census, 2026-10-03

The user subsequently authorized one Runpod B200 for a full pass over the
8,688 training trajectories with unadapted Qwen3.5-9B in vLLM. Resuming
`alzfug70g5237b` failed because its physical host had no free GPU; a separate
live read confirmed that it remained EXITED. New Pod `3422tdq7tfog52`
(`gleipnir-b200-injection-audit`) started in US-NC-2 at the quoted $6.79/hour,
using the existing 200 GB network volume `ixbh81vf9c` at `/workspace` and the
same CUDA-13 image. No additional network volume was created.

Use `scripts/runpod_cloud.py --pod-file .runpod/injection-audit.json` for this
Pod; the previous `.runpod/pod.json` still describes the stopped original Pod.
Refresh the sanitized live snapshot before reconnecting after lifecycle changes.
The new host reports B200 183,359 MiB, driver 595.91.07 and CUDA 13.2. The
preserved environment has Python 3.12.3, PyTorch 2.11.0+cu130, Transformers
5.14.1 and vLLM 0.24.0. CUDA-13 wheel integrity passed, and serving logs
confirmed FlashInfer GDN prefill. This inference check does not validate a
training recipe on the new host.

The campaign and frozen scoring contract are in
[`training_injection_census`](../experiments/training_injection_census/README.md).
The current session has no agent scheduling tool; monitoring occurs in the
active turn and cannot promise a follow-up after yielding. Preserve the volume
and collected artifacts when the user subsequently requests stopping compute.

### B200 stopped after the 9B augmentation replication, 2026-10-04

The user explicitly authorized stopping the existing B200 after the fixed 9B
training/evaluation replication and verified artifact collection. Pod
`3422tdq7tfog52` (`gleipnir-b200-injection-audit`) was stopped; an independent
live read confirmed **EXITED**, null runtime and no direct SSH mapping.
The separate network-volume read confirms `ixbh81vf9c`
(`gleipnir-b200-workspace`, US-NC-2, STANDARD, 200 GB) remains present.
Compute is stopped; retained storage continues billing. Neither Pod nor volume
was terminated or deleted. The older stopped Pod `alzfug70g5237b` was untouched.

All 272 training updates and 15,138 fresh scores completed. Collection verified
all 58 remote artifact files (2,864,290,636 bytes), including the FP32 master,
serving adapter, final checkpoint, raw decision logprobs and logs. Independent
local audits verified exact rebased tensor equality, adapter-specific serving
parity, canonical input coverage, rendered prompts, labels and recomputed metrics.
Exact remote reports were retained before local recomputation changed the
absolute benchmark filename. See the [finding](findings/monitor_injection_augmentation_9b.md).

Before stop, direct SSH and the remote inventory confirmed no remaining GPU
process. `findmnt` confirmed `/workspace` is the network-volume mount. Kernel
links for FLA, causal-conv1d and Triton resolve into `.cache/kernels/`; FlashQLA
is under `.cache/kernels/flashqla-da06429`. The shared training cache resolves to
`.cache/training/student_injection_awareness`. The checked container-local Torch
Inductor, FlashInfer, torch_extensions and Triton cache paths were absent; no
additional compiler-cache archive was needed. Existing runtime-resume archives
and workspace environments remain available. Restore temporary kernel symlinks
and refresh live SSH mappings before a separately authorized future restart.

Sanitized lifecycle reads, storage identity, preservation checks and local audit
hashes are in `results/b200_shutdown/20261004_augmentation_9b/stop_receipt.json`.
Both `.runpod/monitor-injection-pod.json` and `.runpod/injection-audit.json` now
record EXITED; `.runpod/pod.json` still describes the older stopped Pod.

### B200 restarted for the BF16 FA4 screen, 2026-10-04

The user authorized one B200, preferably NC2, to revisit training throughput
after the move from QLoRA to unquantized BF16 LoRA. A live catalog read found
LOW B200 stock in US-NC-2 at $6.79/hour. Pod `9gxht4kafwfbdu`
(`gleipnir-b200-bf16-fa4`) was created using the retained 200 GB network volume
`ixbh81vf9c` at `/workspace`, a 50 GB container disk and the CUDA-13 image
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`, read from the prior
Pod inventory. Direct SSH and `findmnt` confirmed the mount and NVIDIA B200
183,359 MiB, driver 580.126.09. Python 3.12.3, Torch 2.11.0, Transformers 5.14.1
and vLLM 0.24.0 remain present. No additional volume was created.

Use `scripts/runpod_cloud.py --pod-file .runpod/bf16-fa4.json`; refresh its
sanitized live SSH mapping after a restart. Temporary FLA/conv/Triton symlinks
were restored. Shared training caches resolve through `.cache/training/shared`
to `.cache/training/student_injection_awareness`; FA4's existing CuTe cache and
FlashQLA's persistent caches are retained. The matched systems contract is
[`b200_bf16_fa4`](../experiments/b200_bf16_fa4/README.md).
This session has no agent heartbeat scheduler; checks occur in the active turn.
The Pod remains running unless the user separately requests stopping it.

The [FA4 screen](findings/b200_bf16_fa4.md) stopped at eager packing parity;
FA4 performed no optimizer updates. The SDPA control completed, but an overlapping
merge from the retained original 4B compiler cache invalidates a warm-cache timing
comparison. The merge keeps existing shared entries and preserves both source
caches. Failed receipts and exact executed sources remain on the volume.

After the user explicitly accepted the approximately 8% gradient difference, a
separate matched continuation completed both 20-update trajectories. FA4 measured
updates are 20.24% faster with identical partitions/tokens; eager/compiled gradient
differences are recorded as strict failures and accepted separately under 10%.
Full receipts, logs and FP32 adapters are collected locally. The GPU is idle after
the screen; Pod `9gxht4kafwfbdu` remains RUNNING at $6.79/hour. No recurring agent
heartbeat is scheduled and no default recipe change follows from this one screen.

### B200 terminated after attention optimization screens, 2026-10-05

At the user's explicit request to defer further work, Pod `9gxht4kafwfbdu`
(`gleipnir-b200-bf16-fa4`, US-NC-2) was permanently terminated at
2026-10-05 02:34 UTC. The terminate API returned HTTP 204; a subsequent get
returned HTTP 404 / `pod not found`. This supersedes the running status above.
No replacement capacity was launched or scheduled.

Before termination the GPU was idle (0 MiB / 0% utilization), no experiment
process remained, and six final artifact SHA-256 checks passed locally,
including the completed 20-update model metadata, FP32 adapter, full-model
profile and both completed projection pilots. Earlier controls, source archives,
negative receipts and logs had already been collected. The actual shared-cache
mount resolves to `/workspace` on
`mfs#us-nc-2.runpod.net:9421[/networkvolumes/ixbh81vf9c]`.

An independent post-termination volume read confirms the retained STANDARD
200 GB network volume `ixbh81vf9c` (`gleipnir-b200-workspace`) in US-NC-2.
Its project data, environment and persistent caches remain available for a
future authorized Pod; the terminated Pod's container disk is lost. Local
Pod aliases now record TERMINATED. The sanitized lifecycle receipt is
`results/b200_shutdown/20261005_meta_stack/termination_receipt.json`.

Resume with general cuDNN/FROST GEMMs and BF16 MLP fusion, including LoRA
additions and training backward, while retaining BF16 FA4. The
[Meta optimization finding](findings/b200_meta_stack.md) records the completed
negative training screen and faster forward-only projection pilots. No GEMM
training experiment was launched before this shutdown.

### B200 provisioned for MLP/GEMM screens, 2026-10-05

The user explicitly authorized resuming work on MLPs/GEMMs and provisioning
one B200 in NC2. A live Secure Cloud stock read reported LOW availability in
US-NC-2 at $6.79/hour. Pod `i243nsg10usytq` (`gleipnir-b200-mlp-gemm`)
was created at 2026-10-05 11:38 UTC, read back RUNNING, and is reachable by
direct SSH through `.runpod/mlp-gemm.json`. It uses the CUDA-13 image
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`, read from the
existing stopped Pod inventory, a 50 GB container disk and retained 200 GB
network volume `ixbh81vf9c` mounted at `/workspace`.

SSH verifies NVIDIA B200 183,359 MiB, driver 595.91.07,
UUID `GPU-12d4b74b-73ef-7428-b4ec-1fe79d4586b3`, 20.4 CPU quota and
250,999,996,416-byte memory limit. `findmnt` confirms the original NC2
network-volume mount and shared caches. Torch remains 2.11.0+cu130/CUDA 13.0.
Temporary FLA/causal-conv1d/Triton links are restored to the retained overlays;
the saved cuDNN Frontend includes the SwiGLU MLP/FROST interfaces. No package
upgrade, new volume or cold-cache namespace is created.

The bounded experiment contract is
[`b200_mlp_gemm`](../experiments/b200_mlp_gemm/README.md). Keep BF16 FA4 as
the standard; synthetic component timings alone do not promote a recipe.
No in-chat scheduling tool is available: startup/progress checks are performed
during the active turn, with no autonomous follow-up promised after it ends.

### B200 terminated after serving precision screens, 2026-10-08

At the user's explicit request, Pod `qobmmj1weyevg1`
(`gleipnir-b200-cache-free`, EU-RO-1) was permanently terminated at
2026-10-08 03:49:14 UTC. The delete API returned HTTP 204; a subsequent get
returned HTTP 404 / `pod not found`. Local aliases record TERMINATED and no
replacement capacity was launched.

The GPU was idle before collection. All results, logs and inputs are collected
locally; SHA-256 checks pass for 2,004 important artifact files, including the
FP32 adapter. Verified compressed archives preserve compiler/kernel caches,
native overlays (including both container-only symlink targets), staged runtime
bundles and pinned model weights. Container-root compiler caches and package
versions are also saved. Scratch quota and write stalls required an archive
backup; incomplete cache mirrors are not authoritative. Backup checks and the
sanitized lifecycle receipt are under
`results/b200_shutdown/20261008_euro/`.

This Pod's 100 GB `/workspace` was a Pod-attached volume, so termination deletes
it and the 50 GB container disk. Resume from the local backups. The next focus
is vLLM Lens integration; retain the
[current serving selection](decisions/b200_monitor_score_reference.md).

### User-provisioned NC2 B200 refreshed, 2026-10-08

The user allocated Pod `mnmqm5d3eiyvuz` in US-NC-2 with template `a9dk3g7cny`
and the CUDA-13 image above, at $6.79/hour. Live SSH confirms B200 183,359 MiB,
driver 595.91.07 and retained 300 GB network volume `ixbh81vf9c` at `/workspace`.
Use the refreshed sanitized snapshot `.runpod/lens.json`. `PUBLIC_KEY` was empty;
the documented SSH proxy provided access to install the existing project public
key and start SSH in the container without a restart.

The current tracked checkout and serving receipts are synchronized. Prior source
is archived locally and remotely; obsolete source is quarantined under
`results/b200_provisioning/20261008_nc2_refresh/retired_source/`. Adapters and
frozen benchmark inputs pass checksums. The old 0.24 environment is preserved;
isolated 0.31 dependencies and native overlays are staged for the current recipe.
Scheduler/native preflight and FP8 startup preparation pass. The known strict
MXFP8/BF16 diagnostic failure exactly reproduces the earlier error measurements.
Receipts and setup logs are collected locally under `results/b200_provisioning/`
and `logs/runpod/b200_provisioning/`. The Pod is left RUNNING with an idle GPU.
Reconstruct the disposable merged checkpoint before starting a scoring server.

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
