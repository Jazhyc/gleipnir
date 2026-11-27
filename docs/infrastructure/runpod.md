# Runpod

Status, 2026-11-27: the account has no network volumes. EU `cb7bxu71ug` and
NC2 `ixbh81vf9c` were deleted after their unique master adapters were copied
to the Storage Box ([inventory](../findings/operations/runpod_volume_inventory.md)).
Create Pods without a volume, from `gleipnir-serve` or a fresh environment;
the volume procedures below are historical.

On 2026-10-15 the user provided EU-RO-1 B200 Pod `cvjantiismcaod`
($7.99/hour) with network volume `cb7bxu71ug` at `/workspace`. Its template
leaves `PUBLIC_KEY` empty: install the project key and start SSH (after
`ssh-keygen -A`) through the Runpod proxy. The serving stack was restored from
the local 2026-10-07 EU bundle plus the locked 0.31 environment. The previous
checkout is kept as `/workspace/gleipnir-eu-20261004`; see the
[1M-context finding](../findings/serving/b200_context_million.md) for node identity.

At the user's request the Pod was terminated later on 2026-10-15. The GPU was
idle, all session results and logs were collected locally (dry-run check: no
missing files), and container-only compiler caches were archived with a
checksum to both the volume and `results/b200_shutdown/20261015_eu/`. Delete
returned HTTP 204 and the lookup returned 404. Volume `cb7bxu71ug` is retained
with the repository, `.venv-vllm031`, persistent compiler caches and the HF
base. The `/tmp` runtimes and merged checkpoint are reconstructable from the
local bundle, the volume environment and the FP32 master. No replacement
capacity was launched.

Later on 2026-10-15 the user provided a second EU-RO-1 B200 Pod,
`gswwir1daslht0` (snapshot `.runpod/long-context-2.json`), on the same volume.
For the [seed-variation run](../../experiments/teachers/teacher_seed_variation/README.md),
the user authorized the following:
- **Transfer Pod.** RTX PRO 6000 Pod `stl3t4uz49abia` ($2.49/hour,
  `.runpod/nc2-transfer.json`) mounts NC2 volume `ixbh81vf9c`. It was created
  through the Runpod REST API with the project key in `PUBLIC_KEY`.
  At the user's request it was terminated after the copy (delete 204, lookup
  404); NC2 volume `ixbh81vf9c` (300 GB) is retained.
- **EU volume.** `cb7bxu71ug` grew from 100 to 200 GB after the copy hit its
  quota. The size cannot shrink.
- **EU container.** Resized from 50 to 150 GB. The resize restarted the Pod and
  wiped `/tmp`; SSH moved to port 32443. The pre-restart container caches are
  archived in `results/b200_shutdown/20261015_eu_resize/`.

At the user's request, `gswwir1daslht0` was terminated on 2026-10-16:
delete returned HTTP 204 and the lookup 404. Before termination the GPU was
idle and all session results, adapters, configs and logs were collected
locally, with remote and local file counts matching. Container-only caches
are archived on the volume as
`results/b200_shutdown/20261016_eu/container_caches.tar.zst` (SHA-256
`f2dab5b4…`). Both volumes are retained: EU `cb7bxu71ug` (200 GB) and NC2
`ixbh81vf9c` (300 GB). Three older Pods remain EXITED (stopped, not
terminated). The kernel links `.cache/kernels/{fa4,nvidia_mxfp8}` point at the
EU serving runtime.

The EU volume now holds:
- the NC2 runtime archive (SHA-256 `c4d76a28…`, verified);
- the NC2 FLA, causal-conv1d and Triton builds under `.cache/kernels-nc2/`, and
  FlashQLA at `.cache/kernels/flashqla-da06429`;
- the NC2 training compiler cache, `.cache/training/student_injection_awareness`
  (not yet linked as `shared`);
- the SDPA, FP4/FA4 and QAD masters, with hashes verified.

After a restart:
1. Restore the EU serving bundle and stage vLLM 0.31.
2. Extract the archive's `tmp/gleipnir-serving-runtime` to
   `/tmp/gleipnir-training-runtime` (its FA4/MXFP8 overlays differ from the
   EU serving runtime's).
3. Link `/tmp/gleipnir-{qwen35-fla,qwen35-causal-conv1d,triton-3.7.1}` to
   `.cache/kernels-nc2/`.

`logs/runpod/distillation_temperature/restore.sh` scripts these steps and also
unpacks the 2026-10-16 container caches (about 2.5 minutes in total).

On 2026-10-16 the user provided EU-RO-1 B200 Pod `q7io6ui6gwbqo6` ($7.99/hour,
snapshot `.runpod/distillation-temperature.json`). It runs on the same machine
(`2mj1yzztlz17`) and volume, but on GPU `GPU-3d8d6bde…`. As before, its
template leaves `PUBLIC_KEY` empty, so the project key was installed through
the Runpod proxy. At the user's instruction it was terminated on 2026-10-17 at about
03:06 UTC, after all results were pulled; network volume `cb7bxu71ug` is kept.

On 2026-10-17 the user provided EU-RO-1 B200 Pod `ljzun21fdpft92`
($7.99/hour, snapshot `.runpod/qwen38-27b-training.json`) on the same machine
and volume, GPU `GPU-7a2d0c53…`, driver 580.126.20. The project key was
installed through the proxy; at the user's request the container disk was
resized from 50 to 300 GB (SSH moved to port 37846). The runtime is restored
by `logs/runpod/qwen38_27b_training/restore.sh`, and the Qwen3.8-27B weights
are on the container disk at `/root/hf-27b` (lost on termination).

At the user's request it was terminated on 2026-10-18 at about 02:54 UTC
(delete 204, lookup 404; no Pods remain). Before termination the GPU was idle,
all session results, adapters, configs and logs were pulled locally, and the
container-only compiler caches were archived on the volume as
`results/b200_shutdown/20261018_eu/container_caches.tar.zst` (SHA-256
`37ee8670…`). Both volumes are retained (EU `cb7bxu71ug` 200 GB, NC2
`ixbh81vf9c` 300 GB). The 27B base weights and merged checkpoints were on the
container disk; they are re-downloadable and reconstructable from the FP32
master.

Later on 2026-10-18 the user provided EU-RO-1 B200 Pod `wdbgkrnymh3n8h`
($7.99/hour, snapshot `.runpod/qwen38-27b-serving2.json`) on a different
machine (`ods55hcq87v4`, `GPU-4b31de30…`, driver 580.178.04), same volume. At
the user's request the container disk was resized from 50 to 300 GB (SSH moved
to port 19986). The proxy user is the Pod's `machine.podHostId` from the
GraphQL API (`wdbgkrnymh3n8h-64411953`), not a fixed suffix; the proxy answers
"container not found" for a wrong one. `logs/runpod/qwen38_27b_serving/restore.sh`
restores the runtime with the 2026-10-18 container caches and downloads the
27B base to `/root/hf-27b` after `.venv` exists (about 3 minutes);
`experiments/serving/qwen38_27b_serving/remerge.py` rebuilds the merged checkpoint.

On 2026-10-19 the user provided EU-RO-1 B200 Pod `hhja1i6l6de288`
($7.99/hour, snapshot `.runpod/pplx-decider.json`) on machine
`2mj1yzztlz17` (`GPU-58276fb0…`), same volume. At the user's request:
- the container disk was resized from 50 to 300 GB (SSH port 24334);
- it ran the [pplx-decider benchmark](../findings/evaluation/pplx_decider_ood_benchmark.md)
  and the [auxiliary-target](../../experiments/training/aux_supervision/README.md)
  extraction.

The template leaves `sshd` stopped after every restart, so the key and
`sshd` are reinstalled through the proxy each time.

At about 20:10 UTC the host restarted the container on its own. It came back
with driver 595.91.07 (previously 580.126.20), and the GPU fabric stayed
"In Progress". CUDA then failed with error 802, "system not yet
initialized", and Runpod's UI flagged the GPU. At the user's request the Pod
was terminated at about 20:40 UTC (delete 204, lookup 404). Before that, the
failed training receipts were preserved on the volume and the logs pulled.
No B200 was available in EU-RO-1 for an immediate replacement.

Later on 2026-10-19 the user provided US-CA-2 B200 Pod `4kdptgfodt6yz2`
($7.99/hour, snapshot `.runpod/aux-ca2.json`, machine `o66sz0zhpcp0`,
`GPU-84d18db5…`, driver 595.91.07, fabric Completed). It has no network volume
(a mount cannot be added after creation), so the EU volume was copied to its
300 GB container disk at `/workspace`. That copy is lost on termination: pull
results locally as they land. At the user's request the EU volume was read
through a CPU Pod, `01yzkkdopzkfv1` (cpu3c, $0.24/hour, `.runpod/eu-transfer.json`),
because EU-RO-1 had no GPU stock. At the user's request it was terminated after the copy was verified (delete 204, lookup 404), with its temporary transfer key revoked on both Pods; volume `cb7bxu71ug` is retained. Three duplicate CPU Pods created by a launch
loop error were terminated within a minute (delete 204, lookup 404). The copy
streams tar or rsync over SSH from Pod to Pod. Each EU-to-US stream is capped
at roughly 12 MB/s, so total rate scales with the number of streams: 6 tar streams
gave about 75 MB/s, and 12 rsync workers (one per directory) about 140 MB/s. It skipped the old
`gleipnir-eu-20261004` checkout and `.cache/{uv,pplx_decider,qwen27b_generations_ood}`.

At about 01:40 UTC on 2026-10-20 the container became unreachable: Runpod
reported the Pod RUNNING with no runtime or ports, SSH timed out, and the UI
flagged a host problem. The one-line auxiliary arm had just finished training
(about step 232 of 245); its outputs were on the container disk only. The
queued seed average had not started. At the user's request the Pod was then
terminated (delete 204, lookup 404; no Pods remain). The one-line arm's
trained adapter was lost with the container disk and must be retrained. Both
volumes are retained (EU `cb7bxu71ug`, NC2 `ixbh81vf9c`).

Later on 2026-10-20 the user provided an EU-NL-1 B300 Pod, `c2e3gj1pmm1jo2`
($8.99/hour, snapshot `.runpod/seed-average-b300.json`, `GPU-02557ba6…`,
sm_103, 275 GB, driver 595.91.07). It has no network volume. A 2-vCPU
`ubuntu:24.04` CPU Pod on volume `cb7bxu71ug` copied a 45 GB working set to
its container disk in about 3 minutes of transfer, at about 410 MB/s:
- Files over 512 MB went as 256 MB byte ranges over 16 parallel SSH streams.
- The vLLM environment's small files went through 16 parallel tar streams.
- Sizes matched for all 79,588 entries, and SHA-256 for the 7 large files.

The default PyTorch image never started on a 2-vCPU CPU Pod. Both CPU Pods
were terminated afterwards, one by the user. The training and serving stack
ran on sm_103 unchanged (FP32 reference, merge, vLLM 0.31 scoring).
The optimized serving stacks did not: their workers reject kernel receipts
validated on another GPU model (`last-query kernel receipt hardware changed`;
see [optimized gap](../../experiments/serving/optimized_gap/README.md)). At the user's
request the B300 was terminated at about 03:50 UTC (delete 204, lookup 404; no
Pods remain). All artifacts were pulled first, with file sizes verified.

Later on 2026-10-20 the user provided EU-RO-1 B200 Pod `lpjt8h7c2ccygh` on
machine `2mj1yzztlz17` (`GPU-3d8d6bde…`, driver 595.91.07, AMD EPYC 9555;
snapshot `.runpod/optimized-gap-eu.json`), on volume `cb7bxu71ug`. It ran:
- the optimized-serving gap runs for both release candidates;
- the Gleipnir 4B v0.2 QAD;
- the 27B teacher-prompt ID and OOD runs.

Two volume issues:
- **Quota.** The volume hit its 200 GB quota during QAD packaging. The
  campaign-layout copy moved to container disk, leaving about 4 GB free
  ([QAD](../../experiments/quantization/qad_v02/README.md)).
- **Bound sources.** The volume checkout still has the bound pre-PATH-fix
  `src/gleipnir/__init__.py` (`b6acfaad…`) that optimized serving requires;
  the next code sync restores the repository version.

At the user's request the Pod was terminated (delete 204, lookup 404; no
Pods remain). Key artifacts were first verified locally by SHA-256. The
volume is retained.

The latest user-provisioned NC2 B200 `fs334v0u4zysvn` was terminated at the
user's request on 2026-10-13. Network volume `ixbh81vf9c` remains retained;
the [shutdown handoff](../findings/operations/qat_shutdown.md) records verified runtime/cache
preservation and completed experiment artifacts. The old SSH snapshot is marked
terminated. Provision replacement capacity only with fresh authorization and
verify live campaign state before reuse.

For the 2026-10-12 QAT preparation, one B200 was provisioned in US-CA-2
after EU-RO-1 and US-NC-2 catalog reads showed no B200 stock. It was then
terminated at the user's request, including its Pod-attached workspace;
the existing EU/NC2 network volumes remain intact. No training was launched. See the
[provisioning receipt](../findings/operations/b200_qat_provisioning.md) for storage,
runtime qualifications and termination verification. Future capacity should
use NC2 or EU to reuse the existing setup.

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
[`fp4_stability`](../../experiments/fp4_stability/README.md). Restore local
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
[`runpod_gleipnir4b_id`](../../experiments/serving/runpod_gleipnir4b_id/README.md): kernel
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
[B200 finding](../findings/serving/runpod_b200_gleipnir4b_id.md) for numerical agreement,
calibration, measured inference throughput, and the cold-start limitations.

On 2026-10-01, after the B200 training/kernel diagnostics and artifact collection,
the user explicitly requested shutting down compute. Pod `alzfug70g5237b` was
stopped and read back as **EXITED**. Network volume `ixbh81vf9c` remains present;
weights, environments, kernel/compiler caches, datasets and artifacts were
verified on `/workspace` before stopping. Small container-local caches and the
Python runtime were archived there and checksummed locally. GPU compute billing
ended; retained storage still bills. Before a later authorized start, read the
[shutdown and restart record](../findings/systems/b200_fouroversix_training.md#subsequent-b200-shutdown-and-restart-state),
refresh live SSH metadata and restore the temporary kernel links. No Pod or
network volume was deleted.
