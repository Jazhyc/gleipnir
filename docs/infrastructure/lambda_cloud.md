# Lambda Cloud

No Lambda Cloud instance is currently running. On 2026-10-24 the user
authorized one `gpu_1x_h100_pcie` instance (`gleipnir-probe-generalization`,
us-west-3, $3.29/hour, Ubuntu 22.04, driver 570 / CUDA 12.8) for the
[probe study](../../experiments/probes/probe_generalization/README.md) with persistent
filesystem `gleipnir-probes` (`/lambda/nfs/gleipnir-probes`). The locked CUDA-13
Torch needs NVIDIA's `cuda-compat-13-0` package (`/usr/local/cuda-13.0/compat`,
picked up by the bootstrap's runtime file) and the CUDA 13.0 compiler packages
(`cuda-nvcc-13-0`, `cuda-cudart-dev-13-0`, `cuda-cccl-13-0` and the cuBLAS,
cuSPARSE, cuSOLVER and NVRTC dev packages; the full toolkit has unmet GUI
dependencies) with `CUDA_HOME=/usr/local/cuda-13.0` to build causal-conv1d.
NFS sequential reads reach about 800 MB/s but memory-mapped page faults stall;
stage hot data to the 1 TB local disk. At the user's instruction the instance
was terminated on 2026-10-25 after all results (44 files, SHA-256 verified) and
logs were collected. A second, user-authorized `gpu_1x_h100_pcie`
(`gleipnir-probe-ensembles`) ran the post-hoc OOD ensemble scoring on
2026-10-25 (22:15–23:05 UTC, light Torch 2.13 cu129 venv instead of the full
bootstrap) and was terminated after its 8 artifacts were SHA-256 verified.
On 2026-10-26 the user authorized one GPU for the
[attention-map study](../../experiments/probes/probe_attention_maps/README.md). The first
`gpu_1x_h100_pcie` (`gleipnir-probe-maps`) went from booting to unhealthy with
SSH unreachable and was terminated at the user's instruction. The replacement
(`gleipnir-probe-maps2`, 11:45–12:40 UTC) ran the study and was terminated after
its 30 artifacts and logs were SHA-256 verified. us-west-3 capacity was scarce:
none was free for about an hour on each attempt.
A user-authorized `gpu_1x_h100_pcie` (`gleipnir-probe-sleightbench`,
2026-10-26, about an hour, terminated around 15:12 UTC) ran the
[SleightBench probe pass](../../experiments/probes/probe_sleightbench/README.md) and was
terminated after its artifacts were SHA-256 verified. Its image lacked
NVIDIA's apt repository: install `cuda-keyring` (ubuntu2204) before the CUDA
13.0 packages, and set `CUDA_HOME=/usr/local/cuda-13.0` in the runtime file.
On 2026-10-27 the user authorized one `gpu_1x_h100_pcie` for the
[27B probe run](../../experiments/probes/probe_scale_27b/README.md). The first instance
(`gleipnir-probe-scale27b`) sat at 89–92 °C with its clocks thermally throttled
to about 450 MHz (`SW Thermal Slowdown`) and was terminated around 11:46 UTC at
the user's instruction, after its logs were saved. The replacement
(`gleipnir-probe-scale27b-2`, 209.20.156.225, about 11:05 UTC on 2026-10-27 to
01:20 UTC on 2026-10-28) also throttled mildly (85–87 °C, 550–750 MHz, about
4.2K tokens/s of 27B extraction) but finished. It was terminated after its 17
result files and 9 logs were SHA-256 verified. The 221 GB host cannot pin the
27B's 166 GB ID store; the sweep streams it (`--eval-resident none`).
On 2026-10-28 the user authorized a third `gpu_1x_h100_pcie` for the
teacher-prompted 27B run, and chose to wait for us-west-3 capacity so the store
could be kept (no H100 PCIe was free at first; an SXM5 was free only in
us-southeast-1). `gleipnir-probe-scale27b-p` (209.20.156.136) launched on the
first poll at 07:40 UTC, held 76 °C under a 3-minute matmul load
(434 TFLOPS at the 350 W cap, no thermal slowdown) and ran at about 6.2K
tokens/s. It was terminated after its 24 result files and 8 logs were SHA-256
verified (about 21:05 UTC). A fourth user-authorized instance
(`gleipnir-probe-depth`, the only single-GPU type free in us-west-3: an H100
PCIe, 21:22–21:38 UTC) ran the CPU-only last-token depth follow-up in a light
venv (numpy, pandas, scikit-learn, CPU torch) reading the stores from the
filesystem, and was terminated after its 2 results and log were SHA-256
verified.
The pull helper refuses `logs/` paths that resolve outside the repository
(symlinks into the Storage Box mirror); copy such logs over SSH. The
filesystem held the pilot stores (`stores/pilot`, BF16 and FP8, about 303 GB),
the full FP8 stores (`stores/full/{prompted,ntp}`, about 805 GB), the
SleightBench NTP store (`stores/sleightbench/ntp`, 4.5 GB, no text) and the
27B block-31 FP8 stores (`stores/qwen38_27b_l31/full/{ntp,prompted}`, 809 and
810 GB), 2.93 TB in all. At the user's instruction it was deleted through the
API on 2026-10-29 (no instance attached; no Lambda filesystems or instances
remain). The stores are gone; rebuilding one means re-extracting it. Probe
results, banks and per-row logits are kept locally under
`results/probe_generalization/`.

The two-H100
`gleipnir-improvement` instance was terminated on 2026-09-08 following explicit
user authorization and verified artifact collection. See the
[shutdown inventory](../findings/operations/gleipnir_improvement_shutdown_inventory.md).
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
