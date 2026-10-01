# Native Four Over Six MLP training on B200

Date: 2026-10-01. Contract:
[`b200_fouroversix`](../../experiments/b200_fouroversix/README.md).
Status: native kernel canaries passed; both full-model preflights failed the
same-weight compilation gate. Stopped after the requested boundary change.
No recipe promotion.

Resumed work and matched compiler controls are recorded in
[`fp4_training_stability.md`](fp4_training_stability.md). The infrastructure
and stop observations below describe the preceding pilot.

Subsequent infrastructure action: the user authorized stopping the B200 on
2026-10-01. Pod `alzfug70g5237b` was stopped and read back as **EXITED**; network
volume `ixbh81vf9c` is retained. See the shutdown/resume record below.

## Intervention and controls

Use the existing Runpod B200 and selected twelve-checkpoint, selectively compiled,
16,384-padded-token/max-eight adaptive physical batches, with 32 examples per
logical AdamW update. Preserve Qwen3.5-4B revision, frozen data/soft targets,
rank-128/alpha-256 FP32 adapters, SDPA and pinned FLA/causal-conv1d/Triton.
Only MLP bases change precision. Attention retains NF4 storage and BF16 compute
from the optimized recipe. This is frozen-base LoRA, not full-parameter training.

Four Over Six 1.0.5 is an isolated, source-built overlay (10m59s build), with
explicit Triton quantization and CUTLASS native NVFP4 matrix multiplication.
Use original BF16 checkpoint MLP weights, 4/6 MSE scale selection, 2D weight
blocks, nearest forward and stochastic input-gradient quantization. Prepack
frozen forward/transposed weights; keep higher-precision masters/LoRA branches.

Upstream 1.0.5 linear backward asserts physical batch one and computes base-weight
gradients even for frozen bases. The tested local wrapper flattens arbitrary
leading dimensions and computes only the frozen base's input gradient, while
using unchanged upstream native quantization/GEMM functions. No upstream package
patch or higher-precision GEMM fallback is used.

## Native kernel evidence

On NVIDIA B200/SM100, Torch 2.11.0+cu130, native arbitrary-batch forward and
backward passed against FP32 multiplication of decoded FP4 operands:

| Physical batch | Forward relative L2 | Input-gradient relative L2 |
| --- | ---: | ---: |
| 1 | 0.001663 | 0.001640 |
| 2 | 0.001645 | 0.001645 |
| 4 | 0.001661 | 0.001668 |
| 8 | 0.001662 | 0.001657 |

Stochastic backward was finite. The recorded extension is the isolated compiled
`fouroversix/_C.cpython-312-x86_64-linux-gnu.so`; explicit backend selection
prohibits simulated/reference matrix multiplication. Reference decoding is used
only to validate the native arithmetic. Evidence:
`results/b200_fouroversix/kernel_canary.json` and its canary log.

## Initial full-model compilation failure

The original global longest-32 selection loaded successfully with required
FLA and causal-conv1d bindings. Before long-context backward or an optimizer step,
the two-input same-weight eager/compiled loss gate failed: eager **1.352499**,
compiled **1.553454**, a **14.9%** increase. Keep the frozen absolute 0.01 plus
1% relative loss limit. This is not evidence of quantized training convergence,
valid native compiled execution, or a speed result.

The initial wrapper keeps native base operations opaque but permits surrounding
MLP arithmetic to compile. The failure motivates isolating that larger boundary;
the cause has not been established. Artifacts/logs under
`results/b200_fouroversix/` and `logs/runpod/b200_fouroversix/` were collected
locally before follow-up. No optimizer update occurred in the failed campaign.

## Separate eager-MLP boundary diagnostic

`eager_mlp_config.yaml` preserves all numerical gates and longest-row preflight
while keeping entire MLPs eager. Surrounding decoder layers remain selectively
compiled, with the same checkpointing and adaptive batching. The planned controls
compare the original optimized NF4 control, an NF4/eager-MLP boundary control,
BF16/eager MLP bases, and Four Over Six/eager MLP bases. Match initial FP32
adapter hashes, cohort and logical update membership. Ten-step training and before/after training probes
were planned as bounded diagnostics, with no held-out quality selection.

The native arithmetic canary passed again. All 32 MLP interfaces were confirmed
eager, all 96 MLP bases used the native wrapper, and FLA/causal-conv1d bindings
were confirmed. The model gate nevertheless failed with exactly the same values:
eager **1.352499008178711**, compiled **1.5534536838531494**. Expanding the eager
boundary did not resolve the disagreement; this result does not establish its
cause. No long-context backward or optimizer update occurred in either run.

The user asked to stop after trying this change. The launcher was paused while
the current preflight continued, then allowed to record its failure and exit.
No ten-step controls or further variants ran. Both failed campaigns' JSON
reports, contracts and logs were collected locally; follow-up evidence is under
`results/b200_fouroversix_eager_mlp/` and
`logs/runpod/b200_fouroversix_eager_mlp/`. The B200 Pod remains running and idle.
There is no model-training throughput, memory-saving or learning-quality result.

Focused local verification: 43 wrapper/batching/audit/launch tests passed before
the original run; 15 boundary/checkpoint/compile tests passed for the follow-up.
Ruff passed. Native isolated backward is validated; the optimized full-model
training integration remains blocked by the unchanged compilation gate.

## Resume checklist and stability candidates

Handoff recorded 2026-10-01 after the user requested documentation and deferred
further work. The proposals below are unimplemented and untested. No additional
GPU work was launched for this handoff.

### Code, artifacts and infrastructure

- Pilot implementation: commit `b7718cbf1ee6ab52ae5cf297c036a5b83e2ab187` on
  `feat/b200-fouroversix-training`. Executed base revision:
  `213a4db20af7ddbd1dd974ec806577c3f0c8529a`. Each run's `contract.json` records
  the executed file hashes; `commit_receipt.json` identifies subsequent changes.
  Follow-up executable Python sources match the pilot commit; its README was
  updated after execution. Preserve these records when comparing future runs.
- Both `results/b200_fouroversix/` and
  `results/b200_fouroversix_eager_mlp/` are collected locally and retained on the
  Pod. Each contains `contract.json`, `status.json`, `kernel_canary.json`,
  `native-global-preflight_job.json`, `native-global-preflight/screen.json`,
  and `commit_receipt.json`. Logs are in the corresponding
  `logs/runpod/<experiment>/` directory. These ignored artifacts are not in Git.
- Each full-model report records 96 native MLP bases, 960 forward calls and
  **zero backward calls**. The isolated canaries exercised backward separately.
  No FP32 adapter checkpoint was produced by these failed preflights.
- At the last verified completion check, both experiment processes had exited
  and the GPU showed 0 MiB used, 0% utilization and 30 C. The existing B200 Pod
  `alzfug70g5237b` was left running; it was not terminated. This is a historical
  observation, not a promise about its state when work resumes. No recurring
  agent monitoring is active.
- Remote repository: `/workspace/gleipnir`; SSH/transfer helper:
  `scripts/runpod_cloud.py`. Recheck live Pod/SSH metadata and GPU health before
  resuming. Source `.cache-runtime.env`; retain the isolated kernel overlay
  `.cache/kernels/fouroversix-1.0.5`, FLA/conv/Triton targets and persistent
  `.cache/training/qwen35_4b_b200_fa4/gpu-0` compiler cache.
- Recorded stack: Python 3.12, Torch 2.11.0+cu130/CUDA 13.0, Transformers 5.14.1,
  PEFT 0.19.1, Triton 3.7.1, FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1,
  Four Over Six 1.0.5. The overlay installation did not change `uv.lock`.

The selected baseline remains the
[adaptive B200 recipe](../decisions/b200_adaptive_training_recipe.md), with its
historical gradient-parity limitations recorded in the
[execution audit](b200_execution_audit.md). The FP4 pilot did not replace it.
Retain logical batch 32/accumulation 1, adaptive 16,384 padded tokens/max 8,
29,696 context, checkpoint indices
`[0,2,5,8,10,13,16,18,21,24,26,29]`, SDPA,
`full_attention_and_linear_shell`, rank-128/alpha-256 FP32 adapters and the
existing objective/data/optimizer. The one-update global preflight explicitly
uses zero warmup so its planned update would have nonzero LR; ten-step controls
retain the selected scheduler.

### Recipe differences worth testing

Source: humans&' July 2026
[The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl). Its experiments use
Qwen3-30B-A3B MoE RL at 8k context. Transfer to our dense Qwen3.5-4B LoRA
distillation is a hypothesis.

| Candidate | Current pilot | Proposed change and check |
| --- | --- | --- |
| Per-token activation scaling | Four Over Six 1.0.5 Triton defaults to `x.abs().max().float()` over the flattened activation tensor. | Give each token row its own FP32 scale across hidden features, while retaining block scales. Check the same trace alone, with different neighbours/padding and under different physical partitions; check prefix outputs with different suffixes. Audit quantized values/scales and model decision logits. |
| Dequantized BF16 backward | Native FP4 input-gradient GEMM, stochastic gradient rounding, separately prepacked transposed weights. | Compute base input gradients in BF16 using the exact forward quantized weight dequantized to BF16. Validate against that operand's dense input-gradient calculation, then compare adapter gradients and actual clipped AdamW updates. Measure the backward speed/memory cost. |
| BF16 final MLPs | All 32 decoder MLPs use FP4 base GEMMs. | Test zero-based MLP layers 27–31 in BF16 as an initial sensitivity hypothesis. Keep attention and adapter policy matched; record the precise module layout and memory/throughput change. |
| Quantization consistency across paths | Native arithmetic is checked against decoded operands, but whole-model eager/compiled parity fails. | Compare packed values, scales and selected 4/6 branches for identical captured operands across repeated calls and execution paths. Record the first divergence before changing precision or relaxing a numerical gate. |

The [TransformerEngine row-scaled implementation](https://github.com/NVIDIA/TransformerEngine/pull/2931)
provides a concrete per-token scaling reference. Our pinned backend's tensor-wide
maximum makes other rows influence a token's scale; adapting physical batches
can therefore change the quantized function. This has not been measured in our
model and does not by itself explain an eager/compiled comparison on the same
batch. A row-scaled GEMM must apply the matching output correction; changing only
the maximum-reduction axis is insufficient. Inspect native backend support
before choosing an implementation.

The [TransformerEngine backward override](https://github.com/NVIDIA/TransformerEngine/pull/2644)
distinguishes original-master BF16 backward from BF16 backward using dequantized
forward operands. For our frozen bases, only input gradients are required:
`dX = dY @ DQ(W_forward_fp4)`, under the usual straight-through approximation
to activation quantization. No frozen base weight-gradient path is needed.
LoRA branches retain their current precision/autograd behavior. Reusing the
forward quantization decisions avoids an additional mismatch from independently
quantizing a transpose. This remains approximate differentiation through
quantization; it is not an exact derivative of rounding. The current
`NVTE_BACKWARD_OVERRIDE` setting would not affect our custom CUTLASS wrapper;
porting this idea requires code changes or a separately validated TE integration.

The blog retains roughly the final 15% of layers in BF16 and stresses matching
quantization across execution paths. Our five-layer proposal and operand-level
compiler audit are adaptations of those ideas. Its shared-expert advice has no
direct counterpart in this dense backbone. We already use adaptive 4/6 scaling
for weights and activations; enabling it again is not a new intervention.

Four-bit GEMM operands do not imply four-bit residuals, LoRA activations or all
checkpointed tensors. BF16 masters plus packed forward/transposed copies remain
resident. Any activation-memory claim requires measured allocated/reserved
memory at matched batch, rank and context.

### Order of work when resumed

1. Verify the Pod and kernel overlay, freeze a fresh diagnostic contract/output,
   and reproduce the exact existing two-input gate in evaluation mode. Record
   repeatability for fully eager and compiled paths with identical weights,
   inputs, masks, positions, RNG and cache state. Confirm no generation KV cache
   or optimizer update affects the comparison.
2. Locate the first layer/operator where outputs differ. Capture its inputs,
   decision logits and quantized operands/scales. Add matched NF4 and original
   BF16-MLP controls: neither ten-step control ran in the failed campaigns, so
   the existing failure cannot yet be attributed specifically to FP4. Check
   surrounding attention/residual/norm execution and the loss path as well as
   quantization. Complete eager MLPs already failed to remove the discrepancy.
3. After resolving or independently isolating forward parity, test per-token
   scaling and dequantized BF16 backward as separate interventions; then test
   the five BF16 final MLPs. Preserve cohort, initial adapter hashes, update
   membership, optimizer and numerical gates. Record negative results.
4. Only after the relevant forward/kernel/backward gates pass, run the global
   longest-32 memory/gradient preflight and bounded matched update/timing checks.
   Report individual losses, gradient/update comparisons, actual batch
   partitions, native calls, compilation counts, memory and complete-loop time.
   Held-out quality validation remains separate, using the agreed CoT-removed
   ID contract; no final-test selection.

The existing experiment runner automatically queues ten-step conditions after
a successful global preflight. A future forward-only diagnostic needs a bounded
entrypoint and fresh output rather than launching that whole queue implicitly.
Retain the existing compilation loss limit, `0.01 + 0.01 * abs(eager_loss)`;
stochastic backward noise cannot explain a gate that never invokes backward.

## Subsequent B200 shutdown and restart state

On 2026-10-01 the user requested shutting down compute and returning later.
Runpod's stop action succeeded and a separate `get-pod` read confirmed **EXITED**
with `start` available. The Pod was not terminated. A separate volume read
confirmed the 100 GB STANDARD network volume `ixbh81vf9c`
(`gleipnir-b200-workspace`, `US-NC-2`) remains present.

Before stopping, `findmnt` verified `/workspace` is backed by
`mfs#us-nc-2.runpod.net:9421[/networkvolumes/ixbh81vf9c]`. The virtualenv packages,
Hugging Face weights, all four kernel overlays, training compiler caches,
datasets, results and logs are on this mount. The selected compiler-cache link
resolves inside the volume to
`results/b200_training_throughput/compile_cache/h100-recipe-b1`. No GPU process
was active at preservation time. Earlier statements that the Pod was left running
describe the experiment's completion; this shutdown supersedes that state.

Container-local dependencies were also preserved before stopping:

- `/root/.cache/vllm` and `/root/.nv/ComputeCache`;
- `/usr/bin/python3.12`, `/usr/lib/python3.12`, `/usr/include/python3.12`.

Archive: `/workspace/gleipnir/.cache/runtime-resume/container-runtime-20261001.tar.gz`,
31,766,142 bytes, with 1,564 regular files checked against their original SHA-256
digests. Archive SHA-256:
`e80c02715a3dcdf42d8b1a7f870f59b5238f6d53b4f24e291492d92f233c3741`.
A local copy at the same repository-relative path passed the archive checksum.
Local ignored `results/b200_shutdown/preservation.json` records paths,
resolutions, versions and checks; `stop_receipt.json` records the API read-back.
The preservation manifest is also on the volume. The local stop receipt was
written after SSH shutdown, so it is not yet copied back to the Pod.

On a later authorized resume, start the same Pod, verify it reacquired a B200,
and refresh `.runpod/pod.json` from a live read before SSH. That local snapshot
now records EXITED with the old SSH mappings removed. Preserve the existing
CUDA-13 image and verify Python 3.12.3 plus the recorded software/kernel versions
before reusing compilation caches. The virtualenv's interpreter resolves to
`/usr/bin/python3.12`, outside the volume; verify the new container provides it
and use the preserved runtime archive if recovery is needed. Restore additional
container-local caches from the archive as needed, preserving its checksums.

Recreate these missing temporary links after a fresh container starts:

```bash
ln -s /workspace/gleipnir/.cache/kernels/fla /tmp/gleipnir-qwen35-fla
ln -s /workspace/gleipnir/.cache/kernels/causal_conv1d /tmp/gleipnir-qwen35-causal-conv1d
ln -s /workspace/gleipnir/.cache/kernels/triton /tmp/gleipnir-triton-3.7.1
cd /workspace/gleipnir
source .cache-runtime.env
```

Then verify pinned FLA/causal-conv1d and native Four Over Six before model import,
and follow the bounded diagnostic order above. No new capacity or future run
was launched. Stopping ends GPU compute billing; retained storage continues
billing. Runpod documents the container-disk reset and persistent-volume behavior
in [Manage Pods](https://docs.runpod.io/pods/manage-pods#stop-a-pod).
