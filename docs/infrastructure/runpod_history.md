# Runpod machine history

Dated Runpod machine events, newest last. Current setup and procedures are in
[runpod.md](runpod.md).

## B200 stopped after the completed campaigns, 2026-10-03

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

## B200 restarted for the training-injection census, 2026-10-03

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
[`training_injection_census`](../../experiments/injection/training_injection_census/README.md).
The current session has no agent scheduling tool; monitoring occurs in the
active turn and cannot promise a follow-up after yielding. Preserve the volume
and collected artifacts when the user subsequently requests stopping compute.

## B200 stopped after the 9B augmentation replication, 2026-10-04

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
absolute benchmark filename. See the [finding](../findings/injection/monitor_injection_augmentation_9b.md).

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

## B200 restarted for the BF16 FA4 screen, 2026-10-04

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
[`b200_bf16_fa4`](../../experiments/systems/b200_bf16_fa4/README.md).
This session has no agent heartbeat scheduler; checks occur in the active turn.
The Pod remains running unless the user separately requests stopping it.

The [FA4 screen](../findings/systems/b200_bf16_fa4.md) stopped at eager packing parity;
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

## B200 terminated after attention optimization screens, 2026-10-05

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
[Meta optimization finding](../findings/systems/b200_meta_stack.md) records the completed
negative training screen and faster forward-only projection pilots. No GEMM
training experiment was launched before this shutdown.

## B200 provisioned for MLP/GEMM screens, 2026-10-05

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
[`b200_mlp_gemm`](../../experiments/systems/b200_mlp_gemm/README.md). Keep BF16 FA4 as
the standard; synthetic component timings alone do not promote a recipe.
No in-chat scheduling tool is available: startup/progress checks are performed
during the active turn, with no autonomous follow-up promised after it ends.

## B200 terminated after serving precision screens, 2026-10-08

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
[current serving selection](../decisions/serving/b200_monitor_score_reference.md).

## User-provisioned NC2 B200 refreshed, 2026-10-08

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
and `logs/runpod/b200_provisioning/`. Before the shutdown recorded below,
the Pod ran the passing
BF16 0/1 scorer from the
[historical SDPA recipe replay](../findings/injection/b200_augmented_training.md#historical-bf16sdpa-recipe-replay)
at `/tmp/gleipnir-merged/historical-augmented-sdpa-replay02`, with ordinary
parity-gated scope. Its weights differ from historical, and fresh BF16 ID scoring
partly recovers the current FA4 gap. Its
[APPS follow-up](../findings/injection/b200_augmented_training.md#apps-follow-up-for-the-current-bf16sdpa-replay)
completes on the unchanged scorer, with higher injection alarms and lower matched
discrimination than the FA4-trained BF16 control. The earlier bitwise-identical
FA4 repeat
reuses its original scores; historical BF16/optimized diagnostic controls are
preserved. The selected regular checkpoint remains unchanged. Check live
processes and receipts before reuse.
A sanitized live environment snapshot is saved under
`results/b200_environment/20261009_nc2/`, with checksums, package inventories,
loaded module paths, allowlisted process/runtime settings, mounts, Torch build
information and the completed SDPA training/native/parity receipts. The host
runs Ubuntu 24.04.4/kernel 6.8.0-142-generic on Xeon Platinum 8568Y+;
the container sees 192 host threads but has a 20.4-CPU cgroup quota and
233.76 GiB memory limit. Measured GPU memory is 183,359 MiB (SM 10.0), driver
595.91.07, 1,000 W power limit, with CUDA toolkit 13.0.88. The retained network
volume is 300 GB; `df` reports the shared filesystem's capacity, not its quota.
The container root is 50 GB and `/dev/shm` is 117 GB.

| Runtime | Python | Torch | Transformers | Triton | vLLM | FlashInfer | Loaded cuDNN |
|---|---|---|---|---|---|---|---|
| Training/reference | 3.12.3 | 2.11.0+cu130 | 5.14.1 | 3.7.1 overlay | 0.24.0 retained | 0.6.12 | 9.19.0 |
| Current serving | 3.12.3 | 2.13.0+cu130 | 5.14.1 | 3.7.1 | 0.31.0 | 0.7.0.post1 | 9.26.0 |

Training retains FLA 0.5.2, causal-conv1d 1.6.2.post1, FlashQLA
0.1.3+da06429 and FA4 4.0.0b33 overlays; the completed replay uses SDPA full
attention and BF16 MLPs. Shared training caches resolve to the preserved
`student_injection_awareness` tree; serving additionally uses its persistent
vLLM/FlashInfer/cuDNN caches. The serving snapshot includes the exact live
command and model identity: compiled BF16, FlashInfer attention/GDN, causal LAST
cached 0/1 head, context/token budget 32,768, max sequences 128, prefix cache
off, chunked prefill on and the corrected synchronous pooling scheduler.
Loaded library versions and executed training metadata take precedence over
package inventory or fresh-process Torch defaults when reproducing a run.
No secrets, model weights or full process environments are collected.

The eager [Lens research results](../findings/injection/b200_vllm_lens.md) and
[NC2 concurrency measurements](../findings/serving/b200_score_scaling.md#current-fp8-scorer-on-nc2)
remain preserved for compatible future launches.
The accepted merged BF16 checkpoint is reconstructed on ephemeral storage at
`/tmp/gleipnir-merged/fp4-full-training-bf16`. All source and output file hashes
match the accepted artifact; the rebuild receipt is in the refresh directory's
`merge01/`. Original FP32 adapters and pinned base weights remain on the volume.
The inactive `/tmp/gleipnir-merged/fp4-augmented-bf16` checkpoint is checksum-
archived under `results/b200_historical_augmented_id_storage/` on the volume to
make ephemeral room. Its `archive_receipt.json` records exact files and the
original restore path; masters and merge receipts remain preserved.
The inactive historical augmented and previous BF16/FA4 merged checkpoints are
also checksum-archived on the volume under
`results/b200_augmented_seed_repeat_storage/{historical-augmented-bf16,bf16-augmented-fa4}`.
Its two archive receipts preserve all 12 files per checkpoint and their original
ephemeral restore paths. Merged weights remain on the B200; local collection
contains adapters, outputs and receipts.

## NC2 B200 terminated after SDPA controls, 2026-10-09

At the user's explicit request, Pod `mnmqm5d3eiyvuz` was permanently
terminated. Delete returned HTTP 204; the independent read returned HTTP 404
`pod not found`, verified at 04:31:03 UTC. A separate live volume read confirms
`ixbh81vf9c` (`gleipnir-b200-workspace`, 300 GB) remains present. No replacement
capacity was launched. Local aliases are marked TERMINATED.

Before deletion, three ephemeral merge directories were copied to the network
volume and every copied file was checksum-verified. The completed SDPA merge is
under `results/b200_shutdown/20261009_nc2/merged_checkpoints/` as
`historical-augmented-sdpa-replay02`; restore it to its recorded ephemeral path
for compatible reuse. The bitwise-identical FA4 repeat's unfinished merge is
preserved as incomplete evidence, not a usable checkpoint.
Training/serving runtimes, Python and container compiler caches are preserved in
the verified 9,387,527,896-byte `container_runtimes.tar.zst` (159,479 regular
files); the separate source archive preserves Lens/Gigatoken and launcher
contexts (4,563 verified files). Persistent compiler/kernel caches and all
network-volume artifacts remain intact. Serial gzip was interrupted in favor
of parallel zstd; its partial archive and switch receipt are retained.

Local and remote checks verify 1,193 collected files across the SDPA replay,
its APPS follow-up and the FA4 seed repeat. Adapters, results, frozen contracts,
failed receipts and the sanitized environment snapshot are collected locally.
Merged weights and large runtime archives remain exclusively on the network
volume. Preservation/termination receipts and final server logs are under
`results/b200_shutdown/20261009_nc2/`; restore archives and symlink targets before
launching a compatible runtime, then validate the new live environment.

The [next Lens study](../findings/injection/b200_vllm_lens.md#next-intervention-study) targets
the current SDPA-trained adapter's prompt-injection responses. Training
determinism investigation is deferred at the user's request; it remains
unresolved. This records the next task without authorizing new capacity today.

## User-reserved NC2 B200 checked, 2026-10-09

The user reserved Pod `64h2ad4w3nfq2q` (`probable_ivory_skunk`) after the
catalog reported no B200 stock. A live read confirms RUNNING in US-NC-2, using
same template `a9dk3g7cny`, CUDA-13 image and network volume `ixbh81vf9c`.
Proxy SSH verifies the **same physical GPU UUID** as the terminated Pod,
`GPU-36165f94-d010-1b8a-ebe5-d27fe53b9be9`, with unchanged driver 595.91.07,
183,359 MiB VRAM, 1,000 W limit, kernel 6.8.0-142-generic, 20.4-CPU quota and
233.76 GiB RAM limit; zero uncorrected volatile ECC errors.

The fresh container has Python 3.12.3, base Torch 2.9.1+cu130/Triton 3.5.1 and
cuDNN package 9.13.0.50; custom Torch 2.11 training and Torch 2.13/vLLM 0.31
serving runtimes are absent from `/tmp`, as expected after termination.
Both runtime/source archives and durable caches remain on the mounted volume.
No restoration, server launch or experiment occurs during this inspection.
At inspection, direct SSH refused connections; the account SSH proxy worked.
Use the refreshed
sanitized alias `.runpod/sdpa-interventions.json` after enabling direct SSH for
scripted access. The comparison receipt is
`results/b200_provisioning/20261009_resume/environment_comparison.json`.

The user subsequently authorized restoring Lens. Direct SSH is now enabled
without a container restart. The checksum-identical SDPA merge and pinned
vLLM 0.31/Torch 2.13/Lens 1.3.0 serving environment are restored. The
[BF16 eager SDPA Lens baseline](../findings/injection/b200_vllm_lens.md#restored-bf16-lens-baseline-for-the-sdpa-adapter)
passes master/native/functional checks. Its original process was retired
sequentially for the request-local clamp/span/readout extension; the updated
engine passes fresh gates. Following the completed
[injection-direction campaign](../findings/injection/b200_injection_direction.md), the A/B
readout extension passed a sequential restart and
[JudgeDeceiver transfer](../findings/injection/b200_projection_judge.md). The subsequent
[training-example comparison](../findings/injection/b200_training_direction.md) sequentially
serves the pinned base and restores the SDPA engine; that process/frontend
metadata is recorded in
`results/b200_training_clean_direction/clean_direction01/restored_server.json`.
The later [firewall census](../findings/injection/training_firewall_census.md) retires this
Lens engine sequentially for unadapted 9B scoring, then exits cleanly. The GPU
returns to idle, with merged checkpoint and compiler caches preserved. The
subsequent [activation-filter training comparison](../findings/injection/activation_filter_training.md)
restores the checksum-bound Torch 2.11 training runtime and completes both
BF16/SDPA training and BF16 ID/APPS evaluations. FP32 masters stay on the volume;
both new merges stay under `/tmp/gleipnir-merged/`. The matched-control BF16 server
is subsequently retired for the fixed-direction projection follow-up. The
activation-filtered model's BF16 eager Lens server completes the fixed-axis and
model-specific refit evaluations, recorded in
`results/b200_sdpa_lens/activation_ranked20_projection01/server.json`. It is then
retired for [regular-data concept-ablation training](../findings/injection/concept_ablation_training.md).
The new CAFT adapter completes BF16 ID/APPS evaluation. Its compiled BF16 scorer
is then retired for the model-specific direction refit and subsequent original-
axis projection/addition tests. The CAFT Lens process is later retired for
[base-model JudgeDeceiver steering](../findings/injection/base_judge_steering.md); its passing
unadapted base Lens server is recorded in
`results/base_judge_steering/base01/server.json`. FP32 masters and quality artifacts
remain persistent. The unused
matched-random-removal merge is byte-verified in a persistent archive before
reclaiming its ephemeral copy. The earlier Lens
retirement receipt is
`results/b200_vllm031/firewall_census01_before_qwen9b_retired.json`.
The now-terminated Pod's sanitized aliases are `.runpod/sdpa-interventions.json`
and `.runpod/lens.json`. Restore and startup receipts use
`results/b200_provisioning/20261009_lens_restore/` and
`results/b200_sdpa_lens/sdpa01/`, `sdpa02_projection/` and
`sdpa03_judge_projection/`. No dependency upgrade or new capacity.

At the user's request on 2026-10-10, Pod `64h2ad4w3nfq2q` was permanently
terminated after retiring its idle base Lens process and verifying preservation.
Termination returned HTTP 204; an independent Pod read returned HTTP 404
`pod not found`. A separate volume read confirms STANDARD volume `ixbh81vf9c`
(`gleipnir-b200-workspace`, 300 GB, US-NC-2) remains present. Both local aliases
are marked TERMINATED; no replacement capacity was launched.

The new CAFT and activation-ranked-20% merged checkpoints are copied and
checksum-verified under `results/b200_shutdown/20261010_mechanistic/merged_checkpoints/`.
The SDPA merge's earlier persistent copy is reverified. The pinned runtime
archive is reverified and reused; updated Lens/Gigatoken sources and container
compiler caches are separately archive-verified. Masters, frozen contracts,
failed receipts, scores, captures and persistent caches remain on the volume.
Final server logs, preservation checks and termination evidence are collected
locally under `results/b200_shutdown/20261010_mechanistic/`; large merge/runtime
copies remain exclusively on the volume. The
[CAFT release](../findings/injection/concept_ablation_training.md#execution-and-receipts)
and injection-collection membership are independently verified before shutdown.

## User-reserved NC2 B200 for injection detection, 2026-10-11

Live `get-pod` and SSH checks identify `dzhd3xb7um98zj`
(`unemployed_white_wildcat`) in US-NC-2, with volume `ixbh81vf9c` mounted
at `/workspace`. This is a different B200: UUID
`GPU-d1d0c3f4-e2cb-db0c-34ee-2b1d179ea70b`, 183,359 MiB VRAM, driver
580.126.09/CUDA 13.0. The user explicitly authorized installing the existing
repository SSH public key; scripted access uses refreshed sanitized alias
`.runpod/apps-detection.json`. No capacity lifecycle mutation occurred.

The preserved Torch 2.13/vLLM 0.31 runtime, current native tokenizer sources,
compiler caches and checksum-identical SDPA BF16 merge were restored from the
shutdown archives. CUDA execution and both fresh BF16 master/native gates pass.
The [injection-detection experiment](../../experiments/injection/apps_injection_detection/README.md)
records the matched base/SDPA evaluation; hardware/runtime/model bindings are
under `results/apps_injection_detection/presence01/{base,sdpa}/binding.json`.

At the user's request, this Pod was permanently terminated on 2026-10-11 after
verifying both completed experiments, the persistent SDPA merge and runtime
backup, and archiving current container caches. Deletion returned HTTP 204;
an independent read returned HTTP 404. Volume `ixbh81vf9c` remains present
(STANDARD, 300 GB, US-NC-2). `.runpod/apps-detection.json` is marked TERMINATED.
Preservation and termination receipts are collected locally under
`results/b200_shutdown/20261011_injection/`; model/runtime backups and both
experiment result sets remain on the volume. No replacement capacity launched.

On 2026-10-21 the user provided EU-RO-1 B200 Pod `2r9xvvvsx9m60v` ($7.99/hour,
snapshot `.runpod/sleightbench.json`, machine `klzaw5yqymgh`,
`GPU-c70b532c…`, driver 595.91.07) on volume `cb7bxu71ug`.
- **Disk.** At the user's request its container disk was resized from 50 to
  300 GB; the resize moved SSH to port 32814.
- **Restore.** `logs/runpod/sleightbench_eval/restore.sh` restored the
  stack. Volume-only code sync must keep the pinned `src/gleipnir/__init__.py`.
- **Work.** It ran SleightBench `sb01` and built image `0.2-rc1`.
- **Termination.** At the user's request it was terminated the same day
  (delete 204, lookup 404). Beforehand, all artifacts were pulled (dry-run:
  none missing), the GHCR login was removed and the container caches were
  archived to `results/b200_shutdown/20261021_eu/container_caches.tar.zst`
  (SHA-256 `0313da09…`).

The fresh-Pod check of `0.2-rc1` then ran on a user-approved US B200 Pod,
`n35vsu8v8assiz` (`.runpod/image02-check.json`, created through the REST API
from the image, with `PUBLIC_KEY` set and no volume). It also ran SleightBench
`sb02`–`sb04`. At the user's request it was terminated on 2026-10-21 (delete
204, lookup 404) after every result and log was pulled (dry-run: none missing).

For the APPS evaluation of the released monitors, the user authorized US B200
Pod `1al84w5c99wiki` (`.runpod/apps-released.json`, $7.99/hour, machine
`eosnqzirylf7`), created from `gleipnir-serve:0.2` with no volume. Lens 1.3.0
was installed into an overlay at `/tmp/gleipnir-vllm-lens-1.3.0` from the
hash-locked `experiments/injection/b200_vllm_lens/requirements.txt`; it is not part of
the image. After both runs, with all 566 result and log files
verified locally by checksum, it was terminated on 2026-10-21 (delete 204,
lookup 404).
A second user-authorized Pod from the same image, `skhiibc8fu2130`
(`.runpod/apps-base27b.json`), ran the base-27B reference arms and was
terminated after checksum-verified collection (delete 204, lookup 404).

## EU-RO-1 B200 for the hard-label seeds, 2026-10-24

At the user's instruction, Pod `qhvmtrkgjl8rnt` (B200 `GPU-338d6756…`, machine
`2mj1yzztlz17`, $7.99/hour) was created through the REST API on volume
`cb7bxu71ug` with image `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`,
a 150 GB container disk and the project key `~/.ssh/id_ed25519_runpod` in
`PUBLIC_KEY` (SSH worked without the proxy). The first attempts found no
capacity; a retry succeeded after about 12 minutes. The runtime was restored by
`logs/runpod/hard_label_seeds/restore.sh` (the aux_supervision steps with the
`20261021_eu` caches, about 2 minutes). The volume had reached its 200 GB quota
(about 216 GB of blocks), which failed one checkpoint save. At the user's request
the uv caches (`gleipnir/.cache/uv`, `gleipnir-eu-20261004/.cache/uv`) and the
Qwen3.5-9B Hugging Face snapshot were deleted, leaving about 178 GB;
`results/systems_training_scratch/fp32_master.pt` (2.4 GB, an overwritable
systems-screen slot) remains. After the three seeds were scored and every file
was verified locally, the Pod was deleted at the user's instruction (HTTP 204,
lookup 404). Volume `cb7bxu71ug` is retained.

## EU-RO-1 B200 for the optimized-stack probe benchmark, 2026-10-27

At the user's instruction, Pod `8dkyopaxdp4dk0` (B200, machine `2mj1yzztlz17`,
$7.99/hour) was created through the REST API on volume `cb7bxu71ug`, with the
same image, 150 GB container disk and project key as the 2026-10-24 Pod;
capacity was immediate. `logs/runpod/optimized_gap/restore.sh` restored the
serving stack in a few minutes, and the bound `src/gleipnir/__init__.py` was
put back after each code sync. `sync-code` had been copying the local
`.venv-cpu` onto the volume; it now excludes `.venv-*`, and the stray 2.9 GB
copy was deleted. It ran the
[optimized-stack probe benchmark](../../experiments/probes/probe_optimized_serving/README.md).
After all 19 result files were verified locally by SHA-256 and the logs were
pulled, the Pod was deleted at the user's instruction (HTTP 204, lookup 404).
Volume `cb7bxu71ug` is retained, with the vLLM compile cache (11 GB) and the
uncollected 64-row diagnostic activations
(`results/probe_optimized_serving/diag01/full_depth/activations`, 2.6 GB).

## EU-RO-1 B200 and a Lambda H100 for the 27B optimized-stack probes, 2026-10-29

At the user's instruction, Pod `paom6wximf7dw7` (B200, machine
`ods55hcq87v4`, $7.99/hour) was created through the REST API on volume
`cb7bxu71ug`, with the same image and project key as the 2026-10-27 Pod and a
250 GB container disk. EU-RO-1 had no B200 for about 50 minutes; a poller
created it at 11:31 UTC. `logs/runpod/optimized_gap/restore.sh` restored the
stack, the 27B base went to `/root/hf-27b`, and the bound
`src/gleipnir/__init__.py` was put back after each code sync. It ran the
[27B follow-up](../../experiments/probes/probe_optimized_serving/README.md). After 74
result and log files were verified locally by SHA-256, the Pod was deleted at
the user's instruction around 14:58 UTC (HTTP 204, lookup 404). Volume
`cb7bxu71ug` is retained, with the uncollected diagnostic state dumps
(`results/probe_optimized_serving/opt27b01/*/dump`, 10 GB).

The BF16 OOD references ran on a user-authorized Lambda `gpu_1x_h100_pcie`
(`gleipnir-probe-ref27b`, us-west-3, $3.29/hour, 10:52–12:40 UTC) attached to
`gleipnir-probes`, using a light scoring venv. It was terminated at the user's
instruction after its 4 result files and 3 logs were SHA-256 verified.

## EU-RO-1 B200 for the 27B hard-label baseline; volume damage, 2026-11-03

At the user's instruction, Pod `6ucjc9wunbydk8` (B200 `GPU-2ab4adcc…`,
machine `klzaw5yqymgh`, driver 595.91.07, $7.99/hour) was created through the
REST API on volume `cb7bxu71ug` with the usual image, project key and a 300 GB
container disk, after about 25 minutes of polling for capacity. A user-authorized
CPU Pod (`f2rx7i1093saug`, $0.24/hour) mounted the same volume to export
runtimes to the Storage Box (`research/gleipnir/_runpod_eu_volume_20261103/`:
the serving bundle, the two runtime archives and part of the venv, hash-checked)
and was deleted at the user's instruction (HTTP 204, lookup 404).

**The volume has unreadable files.** Reads of 55 files hang for minutes, then
fail with `ENXIO` from both Pods, repeatably: 41 in `.venv-vllm031`, 6 in
TileLang's third-party trees under `.cache/kernels/flashqla-da06429`, one in
the FlashQLA source tree, one FLA `.pyc` and Triton/TileLang entries in
`.cache/training/{qwen35_4b_b200_fa4,b200_training_trim}` (list:
`logs/runpod/qwen38_27b_hard_labels/volume_scan_bad.txt`). The runtime archives
under `results/b200_shutdown/` and the campaign inputs read cleanly and passed
their hashes. Damaged directories were renamed `*.damaged-20261103` (the
unscanned `.cache/{vllm,flashinfer,torchinductor}` `*.unverified-20261103`),
and the Pod links those paths to container-disk rebuilds; see the
[experiment run log](../../experiments/training/qwen38_27b_hard_labels/README.md). Treat
volume contents as unverified until rehashed; rebuild environments rather than
reusing them.

The Pod also rescored the hard checkpoint on the optimized stack. Its
container-only caches (training, vLLM and FlashInfer caches with the warm
cuDNN-frontend cache, the rebuilt FlashQLA target, FLA/causal-conv1d/Triton
kernels and Inductor cache; not the venv) are archived on the Storage Box as
`results/b200_shutdown/20261104_eu_hard27b/container_caches.tar.zst` (SHA-256
`b0e9b6ee…`). At the user's instruction it was deleted on 2026-11-04 (HTTP
204, lookup 404); no Pods remain. Volume `cb7bxu71ug` is retained.

On 2026-11-10 the user authorized a B200 Pod for two report ablations
([4B full specification](../../experiments/evaluation/gleipnir4b_full_specification/README.md),
[filter margins on OOD](../../experiments/teachers/filter_margin_ood/README.md)):
`xaqw5n348zwok3` (`gleipnir-report-ablations`, $7.99/hour, snapshot
`.runpod/report-ablations.json`), US-NC-2 with volume `ixbh81vf9c` at `/nc2`
and the release image `ghcr.io/jazhyc/gleipnir-serve:0.2`. The REST API
rejects `US-NC-2` in `dataCenterIds`; omit the field and attach the volume,
which pins the datacenter. The image's `vllm` console script has a stale
shebang, so start vLLM with `python -m vllm.entrypoints.cli.main`. Results and
logs were pulled and checksum-verified, and the Pod was deleted the same day
(HTTP 204, lookup 404). No Pods remain; both volumes are retained.

The same day a second B200 Pod ran the
[4B serving ladder](../../experiments/serving/gleipnir4b_serving_ladder/README.md):
`x82ld1enklasda` (`gleipnir-4b-serving-ladder`, $7.99/hour, US-CA-2, snapshot
`.runpod/4b-serving-ladder.json`). B200 creation failed with "no instances
currently available" for about 40 minutes in EU-RO-1 and then in every
datacenter (also in the web UI) before one was allocated. A Pod created
without a network volume gets a default 20 GB volume at `/workspace`, which
hides the release image's repository; the API cannot shrink it, so PATCH
`volumeMountPath` to another path (the Pod restarts on the same machine), or
set the mount path at creation. Deleted after verified collection (HTTP 204,
lookup 404); no Pods remain.

On 2026-11-15 the user authorized an EU-RO-1 B200 Pod for the
[serving concurrency and length sweeps](../../experiments/serving/serving_concurrency/README.md):
`nsw57y1tcxh9gw` (`gleipnir-serving-concurrency`, $7.99/hour, no network
volume, `volumeMountPath` `/podvol`, snapshot `.runpod/serving-concurrency.json`).
Creation failed with "no instances currently available" for about 100 minutes
of once-a-minute retries before EU-RO-1 allocated one. Python `urllib`
requests to the REST API were rejected (HTTP 403, error code 1010); `curl`
works. A fresh Pod needs `experiments.release_serving.fetch` for each model
and the benchmark data before a run. It ran from 02:19 to 04:37 UTC and was
deleted after verified collection (HTTP 204, lookup 404); no Pods remain.

The same day the user authorized a second EU-RO-1 B200 Pod for the
[per-change breakdown and 4B graph sweep](../../experiments/serving/serving_concurrency/README.md):
`179xu9pmh64fxh` (`gleipnir-serving-breakdown`, $7.99/hour, same settings,
snapshot `.runpod/serving-breakdown.json`), allocated at 16:47 UTC after
about 70 minutes of retries, on the same machine and GPU as the first. Its
results were collected and verified at 00:22 UTC on 2026-11-16; it then ran
the checks of the image 1.0 release code and was deleted at 01:07 UTC after
verified collection (HTTP 204, lookup 404). At the user's instruction
it was left running for follow-up measurements; check the live inventory
before relying on this.

On 2026-11-16 the user authorized a fresh Pod for the
[image 1.0 check](../../experiments/release_serving/README.md): `7k2awsacb6zxyb`
(`gleipnir-image10-fresh`, US-NC-2 B200, $7.99/hour, from
`gleipnir-serve:1.0-rc1`, snapshot `.runpod/image10-fresh.json`), allocated
at 01:09 UTC after EU-RO-1 had no capacity and deleted at 01:44 UTC after the
check passed and its logs were collected (HTTP 204, lookup 404). For the
generation-mode measurement the user asked for EU-RO-1 hardware:
`8v7naalpvfybi5` (`gleipnir-generation-mode`, $7.99/hour, B200
`GPU-2ab4adcc…`, EPYC 9555, snapshot `.runpod/generation-mode.json`),
allocated at 01:23 UTC on a different machine (`klzaw5yqymgh`) from the
benchmark Pods, and deleted at 03:22 UTC after the measurement finished and
its results were collected (HTTP 204, lookup 404). No Pod remains.

For the [B300 trial](../../experiments/serving/serving_concurrency/README.md) (user
authorization 2026-11-16, any datacenter): no B300 was available at 14:48 UTC
in EU-NL-1, EUR-IS-1 or unrestricted; a retry every 20 minutes allocated
`4sr7r26kfoc0z9` (`gleipnir-b300-trial`, US-WA-2, $8.99/hour,
`GPU-ac47d51d…`, Xeon 6787P, image 1.0, snapshot `.runpod/b300-trial.json`)
at 15:11 UTC on the second try. The image lacks `experiments/serving/gleipnir4b_serving_ladder`
and `data/b200_inference_benchmark`, which the benchmark needs; both were
copied over. It was deleted at 16:30 UTC after its results were collected
(HTTP 204, lookup 404). No Pod remains.

For the [image 1.1 check](../../experiments/release_serving/README.md) (user
authorization 2026-11-16: one B300 and one B200, each deleted after its
check): B300 Pod `ef2mkaw6oa4orn` (`gleipnir-image11-b300`, EUR-IS-1,
$8.99/hour, `GPU-da027ebc…`, Xeon 6740E, from `gleipnir-serve:1.1-rc1`,
snapshot `.runpod/image11-b300.json`) was allocated at 16:46 UTC and
terminated at 17:23 UTC after its logs were collected. `rest.runpod.io` did
not resolve from the VPS at that time, so the termination went through the
GraphQL API (`podTerminate`; `myself.pods` then empty). No B200 was
available from 16:46 to 18:32 UTC (five requests); the user then dropped
the B200 check, so no B200 Pod was launched. No Pod remains.

**API change, 2026-11-19.** `rest.runpod.io` no longer resolves (NXDOMAIN on
public DNS). Lifecycle now goes through GraphQL (`https://api.runpod.io/graphql`,
`podFindAndDeployOnDemand`, `pod`, `podTerminate`, `myself.pods`), which cannot
override an image's entrypoint and has introspection disabled. Images from
`gleipnir-serve:1.2` on start with `/start.sh` as `dockerArgs`.

For the [per-action prefix-caching replay](../../experiments/serving/prefix_caching_replay/README.md)
and the [image 1.2 check](../../experiments/release_serving/README.md) (user
authorization 2026-11-19, one B200 at a time, each deleted after use):
- `5ndq8wcpbn8bmv` (US-TX-6, image 1.1, 22:48 UTC): restart-looped because
  the start command reached the release entrypoint; deleted at 23:09 with
  nothing to collect.
- `ypx6hooyaa38zi` (US-NC-2, `1.2-rc1`, 23:24 UTC; Xeon Platinum 8568Y+):
  image check, a superseded replay and the zstd recompression; deleted at
  00:35 on 2026-11-20 after collection. Its host served the 4B at 141 ms
  per isolated check against 48 ms on EPYC 9555 hosts.
- `2o57cwijinnbp3` (US-CA-2, `1.2-rc2-zstd`, 01:40 UTC; EPYC 9555): rc2
  check and the replay result; deleted at 02:05 after verified collection.

For the [tool-output ablation](../../experiments/evaluation/tool_output_ablation/README.md)
(user authorization 2026-11-20, one B200, deleted after collection):
`9nxfkcv2wrzr24` (US-CA-2, image `1.2`, machine `bipo9o2yngg1`, B200
`GPU-92d608c1…`, Xeon Platinum 8568Y+, driver 595.112.07). Created at 05:58
UTC, reachable over SSH at 06:16 (about 18 minutes to pull the image), run
06:24–06:39, terminated at 06:40 (`podTerminate`; `myself.pods` empty) after
verified collection of 13 files. No Pod remains.

For the [Flint OOD benchmark](../../experiments/evaluation/flint_ood/README.md) (user
authorization 2026-11-20, one B200, EU-RO-1 preferred, terminated after
collection): `8l2ln36loy87kt` (`gleipnir-flint-ood`, EU-RO-1, image `1.2`,
machine `ods55hcq87v4`, B200 `GPU-a2b24934…`, EPYC 9555, driver 595.112.07,
250 GB container disk, no network volume, snapshot `.runpod/flint-ood.json`).
Allocated on the first request at 21:21 UTC and reachable over SSH at 21:28
(the image was cached on the host). It ran an isolated vLLM 0.34.0 runtime
(`/root/flint-vllm034`); the image's 0.31 runtime was not used. Terminated at
22:17 UTC (`podTerminate`; `myself.pods` empty) after verified collection of
25 files; about $7.40. No Pod remains.

For the benchmark's empty-think arm and the Flint agent-trajectory serving
benchmark (user authorization 2026-11-20, one more B200): `j59qldo1255y10`
(same name, settings, machine and GPU; snapshot `.runpod/flint-ood-2.json`).
Allocated on the first request at 22:24 UTC, reachable at 22:25; the compiled
kernels of the first Pod were gone with its disk, so the first engine start
took 581 s. Terminated at 22:49 UTC (`podTerminate`; `myself.pods` empty)
after verified collection of 35 files; about $3.20. No Pod remains.

For the [EvilGenie rescoring on the released stack](../../experiments/adversarial/evilgenie_released_stack/README.md)
(user authorization 2026-11-22, one B200, terminated after collection):
`i7xih3yuodf207` (`gleipnir-evilgenie-rerun`, EU-RO-1, image `1.2`, machine
`2mj1yzztlz17`, B200 `GPU-e5eb9962…`, EPYC 9555, driver 595.112.07, 250 GB
container disk, no volume). Created at 02:30 UTC, reachable at 02:35, run
02:37–02:50, terminated at 02:51 (`podTerminate`; `myself.pods` empty) after
verified collection of 11 files; about $2.80. No Pod remains.

For the [EvilGenie run with assistant messages](../../experiments/adversarial/evilgenie_assistant_messages/README.md)
(user authorization 2026-11-23, one B200, terminated after collection):
`soaw1em7gkhycu` (`gleipnir-evilgenie-messages`, US-TX-6 because EU-RO-1 had
no B200, image `1.2`, machine `40ko8z1ne43c`, B200 `GPU-edea2da8…`, Xeon
6972P, driver 595.112.07, 250 GB container disk, no volume). Created at 04:24
UTC, reachable at 04:40 after a 16-minute image pull, run 04:43–04:58,
terminated at 04:58 (`podTerminate`; `myself.pods` empty) after verified
collection of 11 files; about $4.60. No Pod remains.

For the [Inkling 2 Mini OOD screen](../../experiments/evaluation/inkling_ood/README.md)
(user authorization 2026-11-23: one B200, kept running afterwards for
follow-up experiments): `5uo9tcfq79m724` (`gleipnir-inkling-ood`, US-TX-6
because EU-RO-1 and US-CA-2 had no B200, image `1.2`, machine
`40ko8z1ne43c`, B200 `GPU-62b8cec6…`, Xeon 6972P, driver 595.112.07, 250 GB
container disk, no volume, $7.99/hour, snapshot `.runpod/inkling-ood.json`).
Created at 20:52 UTC, reachable at 21:02. It runs an isolated vLLM nightly
(`/root/inkling-vllm`, `0.34.0rc1.dev1201+gbfe3d7e87`) with the checkpoint in
`/root/hf-inkling` and compile caches in `/root/cache/inkling_ood`; the image's
0.31 runtime is not used. Kept running after the OOD runs at the user's
request for the kernel profile. Terminated at 23:45 UTC on the user's
instruction (`podTerminate`; `myself.pods` empty) after verified collection
of 47 files (`collect.sha256`, `collect_profile.sha256`); about $23. No Pod
remains.

For the [serving stack promotion check](../../experiments/serving/serving_stack_promotion/README.md)
(user authorization 2026-11-27, one B200, any datacenter, terminated after
collection): `x4tbfzde1f4z88` (`gleipnir-stack-promotion`, US-NE-1, image
`1.2`, machine `hl6nl0vylakn`, B200 `GPU-687c5627…`, Intel Xeon 6960P, driver
595.112.07, 250 GB container disk, no network volume). Runpod's Secure Cloud
B200 list price was $8.99/hour that day (Community Cloud $5.98, low stock),
up from the $7.99 of earlier Pods. Allocated on the first request at about
03:55 UTC; terminated at 04:56 UTC (`podTerminate`; lookup null,
`myself.pods` empty) after verified collection of 18 files; about $9. No Pod
remains.

For the [EU volume inventory](../findings/operations/eu_volume_inventory.md)
(user authorization 2026-11-27): CPU Pod `hjk0msbi8ytw7e`
(`gleipnir-volume-inventory-eu`, cpu3c-4-8, $0.12/hour, `ubuntu:24.04`, volume
`cb7bxu71ug` at `/workspace`), created through GraphQL `deployCpuPod` (input
type `deployCpuPodInput`; `dockerArgs` install `sshd` and the project key).
Terminated at 05:41 UTC (`podTerminate`; lookup null, `myself.pods` empty)
after the master adapters were copied and verified; about $0.20. A matching
Pod for the NC2 volume could not be created: US-NC-2 had no CPU instances and
no GPU type at or under $0.50/hour. Both volumes are retained.
