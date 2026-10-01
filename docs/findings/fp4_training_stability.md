# FP4 training stability on the restored B200

Date: 2026-10-01. Experiment:
[`fp4_stability`](../../experiments/fp4_stability/README.md).
This continues the [initial native FP4 pilot](b200_fouroversix_training.md).
No recipe promotion or held-out quality claim follows from forward diagnostics.

## Infrastructure and matched controls

The original B200 Pod `alzfug70g5237b` resumed in US-NC-2 on its original host,
using preserved network volume `ixbh81vf9c`; no data migration was needed.
The B300 fallback and the user's empty reservation Pod are stopped. Actual
hardware: B200/SM100, 183,359 MiB, zero volatile uncorrectable ECC errors.
Python 3.12.3, Torch 2.11.0+cu130, Transformers 5.14.1, PEFT 0.19.1,
FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1, Triton 3.7.1 and Four Over Six
1.0.5 were verified. Model, data, kernel and compiler caches remained intact.

Use the frozen global longest-32 selection and selected 320 training examples,
rank-128/alpha-256 FP32 LoRA masters, the selected twelve checkpoints,
16,384 padded tokens/max-eight adaptive physical batches and logical batch 32.
Attention retains NF4 storage and BF16 compute. Native/BF16 initial hashes match
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
The loss gate remains `abs(eager - compiled) <= 0.01 + 0.01 * abs(eager)`.

## Reproduced forward failure

`results/fp4_forward_diagnostic/` and corresponding Runpod logs were collected
locally. Commit `94f3c3a09f9fc6a0ae089be1381d064d06c80da8` matches the recorded
executed source hashes. No backward or optimizer update ran in either model.

| MLP bases | Eager / eager repeat | Compiled / compiled repeat | Gate |
| --- | ---: | ---: | --- |
| Original BF16 | 1.347804069519043 | 1.347804069519043 | Passed |
| Native FP4, original tensor scaling | 1.352499008178711 | 1.5534536838531494 | Failed |

The FP4 values exactly reproduce the previous failure. Identical repeated losses
show that stochastic backward cannot explain it. The BF16 control passes with
the same data, initial adapters and compiler policy. This narrows the failure
to the precision integration; it does not establish the responsible operator.

Compiling decoder prefixes of 0/1/4/8/16/24/32 layers gives native FP4 losses
1.352499/1.363840/1.506835/1.433760/1.589056/1.690042/1.553454.
The corresponding BF16 losses are
1.347804/1.347804/1.351349/1.351349/1.372149/1.347804/1.347804.
Prefix effects are nonmonotonic, including in BF16; a prefix diagnostic alone
does not identify an operator or demonstrate fallback. Both reports have
unchanged FP32 masters. The FP4 report records 96 native modules, 1,824 native
forward calls and zero backward calls.

## Blog-derived arithmetic interventions

[The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl) motivates per-token
scaling and backward multiplication by the decoded forward quantized weights.
Our implementation is a dense frozen-base LoRA adaptation. It does not use
the blog's MoE/RL stack or its fused row-scaled kernel.

The optional forward normalizes each row by its FP32 maximum, rounds normalized
inputs to BF16, quantizes with fixed tensor maximum one, then rescales the native
CUTLASS output in FP32 and rounds to BF16. This removes cross-token scale
dependence. Backward optionally computes BF16 `dY @ DQ(W_forward_fp4)`;
it avoids a separately quantized transpose. Frozen bases require no weight
gradients. This is a straight-through approximation, not differentiation of
rounding, and the decoded BF16 weight cache adds memory.

On this B200, `results/fp4_native_stability/kernel_canary.json` passes for
physical batches 1/2/4/8: maximum relative L2 0.00235 for native forward versus
decoded operands and 0.00167 for BF16 backward versus FP32 reference arithmetic.
Adding high-amplitude neighbouring rows changes the tested row outputs by zero
relative L2. Counts: seven native forwards, five decoded-BF16 backwards, zero
FP4 backwards. These are kernel checks, separate from model parity, long-context
memory, adapter updates and convergence.

The per-token full-model diagnostic is a separate frozen campaign:
`results/fp4_row_dequantized_diagnostic/`. It reuses the warmed compiler cache
and the completed BF16 control; its configuration records both. The same
initial adapter hash is retained. Eager and eager-repeat loss are
**1.4450643062591553**; compiled and compiled-repeat loss are
**1.434274435043335**. The unchanged gate passes (0.75% relative difference).
No backward or optimizer update ran in this diagnostic.
The BF16 control and row-scaled candidate have identical initial adapters,
example order, probe lengths and original-master common probe losses. Their
initial common-probe mean is 1.3717375844717026; the row-scaled native-probe
mean is 1.4006281197071075. These are training probes, not held-out metrics.

Mixed compiled prefixes of 0/1/4/8/16/24/32 layers give
1.445064/1.398700/1.309612/1.352499/1.386991/1.433421/1.434274.
These remain sensitive to execution boundaries; passing the full-policy gate
does not demonstrate operator-level equivalence. Per-token scaling changes
forward arithmetic; decoded backward has no role in this forward-only result.

`row_dequantized_training.yaml` freezes the next bounded comparison: native
per-token FP4 forward with decoded-weight BF16 backward, original BF16 MLPs,
and optimized NF4 MLPs. Each condition must first complete the separate global
longest-32 backward and one nonzero-LR update. Ten matched updates on the frozen
320-example training selection follow only when its preflight passes. The
compiler's BF16 cast emulation remains disabled in this comparison.

## Successful global-longest training preflight

The native stage completed the global-longest backward preflight and one actual
AdamW update with learning rate 5e-5. The 32 traces contain 922,511 tokens;
the longest is 29,337 tokens. All physical batches are singletons, as required
by the 16,384-token budget for oversized traces. All FP32 adapter gradients
are present and finite. Gradient norm is 26.2123 in the preflight and 26.2148
before the actual norm-one clipped update. Master hash changes from the matched
initial hash to
`b74da89836f96b9eaa3cfc79719bf16440fc13ea60759dcb155ef3393faa46f4`.

The warmed actual update takes 74.5843 seconds. Peak PyTorch allocation is
143,678,906,880 bytes (133.8 GiB), with 144,663,642,112 reserved bytes.
Driver memory snapshots were higher, around 150 GiB during compilation;
allocator peaks do not account for every driver allocation. Cold backward
compilation precedes this timing and is substantial. Do not compare this update
with the old ten-step timing on a different cohort.

Counts are 96 native bases, 10,368 FP4 forward calls, 6,144 decoded-BF16
input-gradient calls and zero FP4 backward calls. Dynamo reports 26 unique
graphs and no unimplemented frames; recorded graph breaks are the intentional
native-base and linear-attention boundaries. The native training-probe mean
improves from 1.400628 to 0.950714; the original-master common-probe mean improves
from 1.371738 to 0.991351. This establishes a finite adapter update and effect
on training probes, not held-out generalization or convergence.

Evidence: `results/fp4_row_dequantized_training/fouroversix-global-preflight/`.
The campaign's executable source hashes match commit
`94d82eb59ee9678cc37f8d8063329e6625f05f87` via its collected commit receipt.

## Matched-cohort gate still fails

The next stage reloads the same original FP32 adapters on the frozen 320-example
selection. Its eager/eager-repeat loss is **1.2848907709121704** and
compiled/compiled-repeat loss is **1.1839299201965332**, a 7.86% difference.
The gate fails before its backward preflight or optimizer updates. The runner
stops the entire campaign; BF16 and NF4 optimizer controls do not run. This
negative result shows that the global-pair gate pass was insufficient evidence
of general compiler consistency. Preserve both reports; do not relax the gate.

The compiler follow-up first keeps original tensor-scaled FP4 arithmetic and
enables the pinned PyTorch option `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`.
The [pinned source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/config.py)
documents preservation of intermediate lower-precision rounding boundaries.
An explicit matched-320 forward-only diagnostic then tests this option with
per-token scaling and decoded backward. Its `steps: 10` sizes the frozen input
selection; `diagnostics_only: true` prohibits all backward and optimizer work.

With original tensor-scaled FP4 and cast preservation enabled, the original
pair passes: eager/eager-repeat **1.352499008178711**, compiled/compiled-repeat
**1.3524880409240723**. This controlled change removes the previously reproduced
14.9% gap on that pair without changing its FP4 scaling. Mixed prefixes remain
variable (0/1/4/8/16/24/32 losses
1.352499/1.253806/1.505310/1.435058/1.274435/1.353217/1.352488), so this is not
an operator-level equivalence result. No backward or optimizer update occurs.
Evidence: `results/fp4_precision_cast_diagnostic/` and corresponding logs.

Cast preservation does not resolve the row-scaled matched-cohort case:
eager/eager-repeat **1.2848907709121704**, compiled/compiled-repeat
**1.3769714832305908**. No backward or optimizer update runs. Evidence:
`results/fp4_row_precision_cast_diagnostic/`. Do not select this option as the
general fix based on its original-pair success.

The `aot_eager` control, preserving per-token native forward and decoded BF16
backward, matches eager loss **1.2848907709121704** exactly on both repeated
calls and all 0/1/4/8/16/24/32 compiled prefixes. Evidence:
`results/fp4_row_aot_diagnostic/`. This narrows the disagreement to Inductor's
generated path in the tested configuration. It is a forward diagnostic, not a
performance benchmark or a training recipe selection.

The next diagnostic observes inputs, normalized quantizer inputs, decoded
quantized activations, row scales and outputs in the first four native MLPs.
It records strides and value differences, with an explicit check that observation
preserves both original losses. Only aggregate metrics are saved; activation
tensors stay in process memory and are discarded when the diagnostic exits.

The first-four-layer trace preserves both original losses exactly. All observed
inputs and quantizer inputs have identical contiguous strides across paths.
The first native gate/up projection already receives different FP32 values:
relative L2 **0.0002089822**, maximum absolute difference **0.01527095**.
After BF16 conversion and row normalization, relative L2 is **0.0006408381**;
decoded FP4 activation relative L2 rises to **0.005456348**. By layer 3's down
projection it reaches **0.2006488**. This measures amplification of upstream
numerical drift, not a stride reinterpretation in the observed tensors.
Evidence: `results/fp4_row_operand_diagnostic/`.

The targeted follow-up keeps Qwen RMSNorm interfaces eager and preserves BF16
rounding in the remaining Inductor operations, while retaining native MLPs,
checkpointing, adaptive batching and the selected compiler policy. It records
all exact normalization module names and repeats the native operand trace.
RMSNorm's contribution remains a hypothesis until this intervention is measured.

Keeping all 81 plain Qwen RMSNorm interfaces eager while preserving casts gives
eager loss **1.2848907709121704** and compiled loss **1.2379920482635498** on the
matched-cohort pair. The original gate still fails; this is not a general fix.
The operand observer preserves both losses exactly and all observed tensors in
the first four MLPs match exactly, including decoded FP4 activations. Compiled
prefixes 0/1/4/8 also match eager exactly. Prefixes 16/24/32 give
1.237992/1.294601/1.237992. Thus the intervention removes the observed early
drift but leaves later differences; it does not locate their exact operator.
Evidence: `results/fp4_row_eager_norm_diagnostic/`, collected locally.

The next bounded training test uses `aot_eager`, which previously matched eager
exactly, for all three MLP precision conditions. It preserves the twelve
checkpointed layers, adaptive batches, frozen inputs and FP32 master adapters.
Each condition must pass its global-longest update before ten matched updates.
This isolates practical training viability from the unresolved Inductor
disagreement. It does not establish the original optimized recipe's performance.

## AOT training campaign

`results/fp4_row_aot_training/contract.json` matches all six executable/config
hashes in commit `21e8cb5`; the verified receipt is saved locally and on the
persistent volume. Native kernel canaries passed again before model loading.

The native global-longest stage completes its backward preflight and one update
at learning rate 5e-5. Repeated eager/compiled losses all equal
**1.4450643062591553**. The update processes 922,511 tokens, has finite gradient
norm **25.28337479**, takes **85.7544 seconds**, and peaks at
**147,515,253,248 allocated bytes** (137.4 GiB). FP32 master hash changes from
the shared initial hash to
`66ae8c473153e743aba99828a5e9720e49cad637d850ebe4e60fe4690c5a999d`.
The common training-probe mean decreases **1.371738 -> 0.961143**; native probe
decreases **1.400628 -> 1.003383**. Counts: 96 native modules, 10,368 native
forward calls, 6,144 decoded-BF16 backward calls and zero FP4 backward calls.
Dynamo records 25 unique graphs and no unsupported operations. The report and
FP32 master checkpoint have been collected locally.

The matched-cohort eager/compiled/repeat losses all equal
**1.2848907709121704**. Its separate longest-32 backward also passes: 711,225
tokens, longest 28,733, finite nonzero norm **25.73112488**, peak allocated
**144,567,890,432 bytes**. It compiles fresh FLA/Triton kernels for new sequence
shapes before optimizer updates. Record this cold work separately from measured
warm-step throughput; this first native pass cannot support a matched speedup
claim against subsequent controls that reuse its kernel cache.

The native ten-step trajectory completes all ten steps, with nine nonzero-LR
updates after the recorded zero-LR warmup step. Every adapter gradient remains
finite and physical batch sizes include 1/2/4/8. The steps process **1,314,331
actual tokens** and peak at **145,166,090,240 allocated bytes** (135.2 GiB).
The common training probe decreases **1.165299 -> 0.779105**; the native probe
decreases **1.205408 -> 0.828045**. The final FP32 master hash is
`907c7a4550773790705437751baeb9cfcedd2a2daa4cd4adb43daa279d5b2f55`.
The report and FP32 checkpoint have been collected locally.

Counts: 96 native bases, 21,324 native forward calls, 14,112 decoded-BF16
backward calls and zero FP4 backward calls. Dynamo records 33 unique graphs and
no unsupported operations. Complete loop time is **495.593 seconds**, including
cold work. Individual step times are
275.68/74.14/14.86/14.74/20.77/16.74/14.67/21.26/30.07/12.50 seconds.
The first two steps dominate cold compilation; subsequent steps vary in tokens
and cache work, so they are not interchangeable performance samples.

Matched BF16/NF4 controls are pending. These updates establish bounded
long-context LoRA training viability, not convergence, held-out quality, full
parameter FP4 training or an Inductor recipe fix. Native MLP forward operands
are W4A4; decoded backward, attention compute and stored residual/checkpoint
activations remain higher precision.

During the first native trajectory, a recorded cache helper reused five missing
complete native Triton entries from the preserved same-B200 cache at
`.cache/training/qwen35_4b_b200_fa4/gpu-0/triton`. It excluded Inductor-generated
groups and incomplete groups, preserved existing entries, and atomically
published copied directories with no replacement. The helper and copied keys
are saved in `results/fp4_row_aot_training/cache_seed_receipt.json` and the
associated ignored script. This is cache reuse, not a kernel source change.
Initial native step times include cold compilation and this intervention;
do not compare their aggregate time against warm controls as a precision speedup.

## Interpreting training memory

The stable native wrapper retains the original BF16 MLP weights for reference
probes, packed FP4 forward weights/scales, and a BF16 cache decoded from those
packed weights for backward. The BF16 MLP control retains only the original
MLP weights. This makes the native path's resident weight budget larger in the
current implementation. The packed upstream `QuantizedTensor` contains values,
scales and amax; it does not contain another hidden master-weight tensor.

W4A4 describes forward GEMM operands. It does not imply four-bit storage of
saved residuals, checkpoint activations, LoRA activations, or optimizer states.
Per-token normalization/rescaling and different compiler boundaries also change
temporary allocations. The extra weight cache explains only part of the measured
FP4/BF16 peak gap; its full attribution requires a saved-tensor/allocator profile.

Compute dtype also does not determine stored activation dtype. Pinned
[bitsandbytes 0.50.0](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.0/bitsandbytes/nn/modules.py#L626-L637)
casts the four-bit linear result back to its input dtype, and
[PEFT 0.19.1](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/tuners/lora/layer.py)
preserves the base result dtype after adding LoRA. Thus FP32 normalization outputs
can propagate FP32 MLP activations through an NF4/BF16-compute path. This is a
relevant precision mechanism, not a measured attribution of the entire memory
gap. Do not infer memory savings solely from weight bits or BF16 compute settings.

## NF4 initialization audit

The initial three-condition AOT campaign finishes every global update and all ten
cohort steps. Its NF4 condition is **not a matched initialization control**:
it starts from hash
`e2944ee2eb7be34c86ccc0e82e75ac282158cabc101258c2db15ea5eed432a49`,
despite the explicit seed zero. Parameter names and ordering match exactly, so
this is a value difference. Native and BF16 match on initialization, example
order, token counts, physical partitions, learning rates and initial common
probe values. Preserve the original NF4 result as an unmatched finite-trajectory
record, not as a paired precision comparison.

The correction uses a standard PEFT initial adapter artifact and checks its
expected full tensor hash before any screen model/GPU work. File checksums enter
the execution contract. The artifact exporter uses the native one-update,
zero-weight-decay checkpoint: reset B to its initial zero and retain A, but
require the complete recovered tensor hash to equal the original initial hash
before export. This exact hash check is mandatory; no approximate inversion or
unverified assumption about unchanged A is accepted.

`row_aot_nf4_matched.yaml` repeats the separate global update and ten cohort steps
with the shared initialization. No native/BF16 arithmetic changes are needed;
their completed trajectories remain valid controls. The corrected NF4 result is
pending. The new guard is tested to fail before model calls or optimizer creation
in both diagnostic and training modes.
