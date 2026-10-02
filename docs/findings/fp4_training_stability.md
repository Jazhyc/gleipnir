# FP4 training stability on the restored B200

Date: 2026-10-01. Experiment:
[`fp4_stability`](../../experiments/fp4_stability/README.md).
This continues the [initial native FP4 pilot](b200_fouroversix_training.md).
No recipe promotion or held-out quality claim follows from forward diagnostics.

Outcome: native W4A4 MLP forward with per-token scaling and decoded-weight BF16
backward completes bounded long-context LoRA updates using `aot_eager`, retaining
the selected partial checkpoints and adaptive batches. Inductor consistency
remains unresolved. This prototype establishes training viability, not a memory
or throughput improvement over the selected optimized recipe.

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

Matched BF16 and corrected NF4 controls also complete; see the final comparison.
These updates establish bounded
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
expected full tensor hash before screen probes, backward or optimizer work.
File checksums enter the execution contract. The artifact exporter uses the native
one-update,
zero-weight-decay checkpoint: reset B to its initial zero and retain A, but
require the complete recovered tensor hash to equal the original initial hash
before export. This exact hash check is mandatory; no approximate inversion or
unverified assumption about unchanged A is accepted.

`row_aot_nf4_matched.yaml` repeats the separate global update and ten cohort steps
with the shared initialization. No native/BF16 arithmetic changes are needed;
their completed trajectories remain valid controls. The corrected NF4 result
completes both stages. The new guard is tested to fail before model calls or
optimizer creation in both diagnostic and training modes.

## Completed matched comparison

`results/fp4_row_aot_training/analysis.json` selects native/BF16 from that campaign
and NF4 from `results/fp4_row_aot_nf4_matched/`. It explicitly excludes the first
unmatched NF4 trajectory. All conditions have the same initial master tensor
hash, trainable names, GPU/software, example permutation, probe inputs, adaptive
policy, per-step token counts/physical partitions and learning rates. Each passes
its global-longest update and ten cohort steps, including nine nonzero-LR cohort
updates. Each cohort processes 1,314,331 actual tokens. All gradients are finite
and nonzero in aggregate, every final master hash changes, and unsupported
compiler-operation counters are empty.

| MLP path | Global update peak GiB | Cohort peak GiB | Training probe before -> after | Loop seconds including cache work |
| --- | ---: | ---: | ---: | ---: |
| Native W4A4 / decoded BF16 backward | 137.4 | 135.2 | 1.165299 -> 0.779105 | 495.593 |
| Original BF16 | 118.6 | 116.7 | 1.165299 -> 0.825084 | 152.886 |
| NF4 storage / BF16 compute | 137.1 | 134.8 | 1.246726 -> 0.937017 | 174.937 |

Memory is whole-model peak **allocated** memory, not driver/process or reserved
memory. All conditions retain NF4 attention storage and BF16 attention compute.
The native common probe uses original BF16 MLP masters; BF16 uses those same
masters, while NF4 retains its NF4 MLP bases. Thus NF4 probe scores also reflect
different base arithmetic; do not interpret this table as held-out quality
ranking or serving parity. None of these runs accesses the final test set.
The first native trajectory pays substantial cold compilation and recorded cache
seeding; no matched throughput gain follows from the loop times. A future systems
claim needs a cached repeat, and quality selection needs grouped held-out tests.

Corrected NF4 final master hash:
`fa1d82481df532983901992fd00de09d4f695fc2c1fb45d852f09e0f2deb4d6c`.
Its global master hash:
`96c8c6602d6564c27d97f881bcc943b17aa634d882abeb0c3b33122feee359bc`.
Its six source/config hashes match commit `376fb78`; its initial adapter file
checksums match the exact recovery receipt. Reports, FP32 checkpoints, logs,
initialization artifacts and receipts are preserved on the network volume and
collected locally. Twenty focused tests, Ruff and diff checks pass. The B200 is
left running and idle; no additional training is queued.

For another FP4 experiment, retain native CUTLASS/Triton, MSE 4/6 weight scaling,
`row_scaled_activations: true`, `backward_mode: dequantized_bf16`,
`compile_backend: aot_eager`, the twelve checkpoint indices and adaptive
16,384-token/max-eight policy. Use a fresh output path and an explicit shared
initial PEFT adapter plus its expected full master hash for every precision
comparison. Preserve original failed Inductor recipes for reproduction; the
user-selected default NF4/Inductor recipe is not changed by this finding.

## Warmed FP4 timing benchmark

The user requested a proper timing estimate, then restricted this benchmark to
FP4 only. `row_aot_timing.yaml` retains the native per-token W4A4 forward,
decoded-weight BF16 backward, exact initial adapters, AOT backend, twelve
checkpoints and adaptive batching. Native canaries and both global/cohort
eager-AOT gates pass; each eager/compiled repeated loss matches exactly. The
global-longest preflight includes an actual nonzero-LR update before timing.

In one model process, warm all ten frozen logical batches once, then measure
three complete replays. Each pass restores the same initial master hash and
creates fresh AdamW state and a fresh ten-step scheduler. Every pass uses the
same membership, ordering, token counts, padding, physical partitions and LR
sequence, with one LR-zero step and nine nonzero-LR updates. This repeats
1,314,331 actual tokens per pass; it does not sample thirty distinct batches.
Final adapter hashes differ across replays, so matching initialization and work
must not be described as bitwise deterministic training trajectories.

| Measured replay | Mean seconds per step | New Dynamo graphs |
| --- | ---: | ---: |
| 1 | 15.541 | 0 |
| 2 | 15.897 | 0 |
| 3 | 15.600 | 0 |

Across all thirty measured steps, mean time is **15.679 s**, median **15.440 s**,
range **11.394–20.153 s**, and token-weighted throughput **8,382.6 actual
tokens/s** (2.041 examples/s). The full measured step work totals 470.380 s.
Batch lengths explain much of the range; same-batch means are retained in the
report. Measured peak allocated memory is **135.196 GiB**, reserved
**137.430 GiB**. All losses and adapter gradients are finite, aggregate gradients
are nonzero, adapters change after every replay, and native forward/decoded-BF16
backward calls are exercised with zero FP4 backward calls.

The separate warm-up pass takes 177.174 s of step work, including first steps
35.357/17.505 s and eight new graphs. Total graphs remain 33 throughout all
measured passes. Existing disk caches were preserved, so this is process warm-up
with reused caches, not a pristine-cache compile-time benchmark. It supersedes
the earlier steps-3-to-10 estimate of 18.2 s for estimating warmed work on this
cohort. Do not infer a pure compile-time difference from total pass times.

CUDA synchronization brackets forward, backward, finite-gradient checks,
clipping, AdamW and scheduling. Report I/O, model loading, numerical gates,
preflight, state restoration, probes and export are excluded. Fresh optimizer
allocation is included once per ten-step measured pass. These are instrumented
training-step timings for the stable AOT prototype, not end-to-end campaign or
epoch ETAs, an Inductor timing, or evidence of a speedup over another precision.

`results/fp4_row_aot_timing/analysis.json` verifies matching workload/initialization,
all numerical gates and the six executable/config hashes against commit
`44e8b2d`. The launch receipt records its parent `9e84c94` plus those exact source
hashes because the scoped feature commit followed successful startup. Reports,
logs and FP32 checkpoints remain on persistent storage and are collected locally.
Twenty-three focused CPU tests, Ruff and diff checks pass. The benchmark has
completed; the B200 is running idle with no further training queued.

## Warmed FP4 overhead profile

`row_aot_profile.yaml` retains the stable unfused AOT recipe, warms all ten
batches and profiles batch five's forward/backward/finite checks/clipping.
Both numerical gates match eager exactly, both longest-input preflights pass,
and the ten warm-up updates complete. The profiled backward leaves the adapter
hash unchanged; it performs no optimizer update. Batch five contains 188,965
actual tokens, 202,576 padded tokens and thirteen adaptive physical batches.

The CPU/CUDA trace records 149,478 kernel launches and 18.352 seconds of kernel
activity. This is an instrumented diagnostic, not a new step-time estimate:
profiler wall time is 28.560 seconds and the corresponding ordinary warm-up
step takes 20.041 seconds. The trace includes GPU user annotations. Exclude
those ranges and do not sum CPU operator attribution with GPU kernel events;
the original raw `profile.json` lists both.

Attributing recorded GPU kernels/copies to their enclosing CPU function scopes
assigns 3.451 seconds to native FP4 forward, 0.643 seconds to decoded BF16
backward, 3.942 seconds to FLA functions and 10.637 seconds to the remaining
work. Inside native FP4 forward, the 1,716 CUTLASS matmuls total 0.256 seconds
and quantization totals 0.560 seconds. Most remaining activity in that scope is
casts, division, multiplication and row reduction. Recorded durations include
profiler effects and do not predict an additive step saving, but support testing
fused row scaling and better compilation around the native GEMMs.

The isolated fused-row CUDA canary passes **bitwise** normalization, FP32-scale
and rescaling comparisons on shapes `(1,256)`, `(128,2560)`, `(513,9216)`,
`(16384,2560)` and `(256,9216)`, with zero and outlier rows. Full model gates and
matched training timing remain separate requirements. The new implementation
preserves FP32 division rounding and BF16 boundaries; it does not fuse the
Four Over Six quantizer or change native GEMM/backward arithmetic.

Reports, trace, FP32 checkpoints and logs are collected under
`results/fp4_row_aot_profile/` and `logs/runpod/fp4_row_aot_profile/` and remain
on persistent storage. The collected `analysis.json` excludes annotations and
verifies all seven source/config hashes against `21bab07`; the launch receipt
records parent `abf9954` plus those hashes. The isolated canary is preserved as
`results/fp4_fused_row_canary.json`. Optimization candidates are predeclared in
the experiment README; no new precision control or held-out quality run is
included in this overhead investigation.

## Inductor boundaries passing forward parity

`row_inductor_boundaries_diagnostic.yaml` preserves BF16 intermediate casts,
keeps the 81 plain RMSNorm interfaces and 32 MLP activation interfaces eager,
and uses `decoder_shells_without_token_mixers`: both gated-delta and full
attention token mixers remain eager while decoder shells compile with Inductor.
Original per-token scaling and native quantization arithmetic remain unchanged.

Eager, repeated eager, Inductor and repeated Inductor canary losses all equal
**1.2848907709121704**. Every compiled prefix (0, 1, 4, 8, 16, 24 and 32 layers)
also gives exactly that value. The initial master hash is unchanged and there
are zero optimizer updates. This passes a previously failing full-model forward
check with narrower boundaries; it does not identify which additional fence
resolved the discrepancy or establish identical training gradients.

The receipt in `results/fp4_row_inductor_boundaries_diagnostic/analysis.json`
verifies all ten source/config hashes against `9a3133a`; its launch revision
records parent `21bab07`. Reports and logs are collected and persist remotely.
The predeclared follow-up `row_inductor_fused_timing.yaml` combines these eager
boundaries with fused row scaling and clip-norm gradient validation, retaining
all global/cohort gates and the original FP4 warmed three-replay protocol.

## Faster warmed FP4 training

`row_inductor_fused_timing.yaml` completes the global-longest adapter update,
ten warm-up steps and three matched ten-step replays on the existing B200.
Both global/cohort eager and compiled repeated losses match exactly. All
adapter gradients pass the missing-gradient and finite, nonzero aggregate-norm
checks; every replay changes the FP32 master adapters. Native W4A4 forward and
decoded-weight BF16 backward remain exercised, with zero FP4 backward calls.

The intervention combines fused normalization/rescaling, cast-preserving
Inductor with the passing eager boundaries above, and `gradient_validation:
clip_norm`. The latter retains `clip_grad_norm_(error_if_nonfinite=True)` before
every update and explicitly checks missing gradients, avoiding redundant
individual finite-check host waits. It does not disable nonfinite rejection.

| Measured replay | Original AOT seconds/step | Optimized Inductor seconds/step | New optimized graphs |
| --- | ---: | ---: | ---: |
| 1 | 15.541 | 12.368 | 0 |
| 2 | 15.897 | 12.566 | 0 |
| 3 | 15.600 | 12.503 | 0 |

All thirty measured steps average **12.479 s**, median **12.258 s**, range
**9.379–16.348 s**. Against the collected original FP4 benchmark's 15.679 s,
this is **20.411% lower step time**, **1.256 times throughput**, and **10,532.3
actual tokens/s** (2.564 examples/s). Every same-batch mean improves by
18.258–22.073%. Total measured step work falls from 470.380 to 374.371 seconds.
The predeclared 5% improvement criterion passes. This combined intervention
does not isolate the gain from each change, and does not benchmark another
precision or establish a whole-epoch speedup.

The collected audit verifies identical source inputs, GPU/software, initial
adapters, trainable names, example permutation, every physical partition,
actual/padded tokens and LR sequence against the original FP4 benchmark. The
three passes replay the same ten batches; final adapter hashes differ between
replays, so bitwise gradient or trajectory parity is not claimed. No held-out
quality or serving comparison is performed.

Measured peak allocated memory is **138.638 GiB**, reserved **143.289 GiB**,
compared with original AOT 135.196/137.430 GiB. This is a speed improvement with
higher peak memory. Original BF16 reference weights, packed FP4 weights and the
decoded BF16 backward cache remain resident; saved activations remain 16/32-bit.
The warm-up pass takes **190.187 s** of step work versus original 177.174 s,
with six new Dynamo graphs. Total graphs then stay at 24 through measured
passes, versus original 33. Graph counts do not measure all kernel warm-up work;
no cold-start or pure compilation-time improvement is asserted. Timed scope
still includes forward/backward, clipping, finite rejection, AdamW and scheduling,
and excludes loading, preflight, resets, probes, report I/O and export.

Use this configuration for the next bounded FP4 trial, retaining fresh output
paths, the explicit initial adapter/hash, both longest-input gates, checkpoint
and batching policies, native canaries and complete warmed replay. The original
user-selected NF4 recipe and historical failed Inductor conditions are preserved.
The separate fused-AOT candidate was not launched because this Inductor
configuration passed the declared performance gate.

`results/fp4_row_inductor_fused_timing/analysis.json` verifies all ten source/config
hashes against `9212d27`; the launcher records parent `9a3133a` plus those exact
hashes. Reports, native/row canaries, logs and both FP32 checkpoints are collected
locally and remain on the persistent network volume. Collected checkpoint SHA-256
values match the remote copies. Thirty focused CPU tests, Ruff, configuration
validation and diff checks pass. The campaign is complete; the B200 is running
idle and no further experiment is queued.

## Precision scope of the 4-bitter Lesson

The [blog's final recipe](https://humansand.ai/blog/nvfp4-rl) is mixed precision:
NVFP4 weights and activations in expert forward GEMMs; BF16 backward GEMMs
using decoded forward operands; higher precision for non-expert components,
shared experts and approximately the final 15% of layers. It updates base
weights in MoE RL, whereas our dense Qwen3.5 experiment updates LoRA adapters.
The blog's 97% expert-parameter illustration is explicitly for DeepSeek-V3-style
models; it does not measure our model's FP4 coverage or the fraction of backward
work performed in FP4.

Our current recipe already has per-token activation scaling, MSE 4/6 for both
operands and decoded-forward-weight BF16 input gradients. Frozen base weights
need no weight gradients, so retaining decoded activations for their weight
gradients would not help. LoRA branches differentiate their own higher-precision
forward inputs. This distinction matters when adapting the blog's backward rule.

The practical remaining gaps are fused activation packing, the faster 4/6
selection kernel, and avoiding redundant resident or saved representations.
The experimental packing canary in the experiment README tests the first gap
without broadening quantization or changing the training arithmetic. Native FP4
backward is not required to follow the blog's final recipe. Selective BF16 final
layers are a separate possible stability experiment, not an implemented policy.

## Tiled activation packing follow-up

The one-program-per-row normalization/packing candidate passes small cases but
fails exact packed parity at `(513, 9216)`. Prohibiting floating-point fusion
also fails: four block-scale bytes and 29 packed-value bytes differ, despite
exact row maxima. Both variants are rejected and their sources/receipts remain
under `results/fp4_packing_kernel_canary/` and
`results/fp4_packing_no_fma_canary/`. Keeping the original quantization tile shape
resolves the observed mismatch; these results do not isolate its compiler cause.

The passing tiled variant computes row maxima separately, then fuses FP32
normalization, the BF16 rounding boundary and upstream 4/6 selection/packing
in the pinned 16x64 TMA tiles. It directly supplies the packed activation to
CUTLASS, eliminating the intermediate normalized BF16 matrix. Rescaling and
decoded-weight BF16 backward remain unchanged. All seven isolated cases match
row maxima, FP4 values, FP8 scales including padding and rescaled GEMM outputs
exactly. Initial warmed call timings improve 1.12–1.62x at the model widths;
these event timings include host dispatch gaps, not just GPU kernel duration.

`row_inductor_packed_timing.yaml` retains the preferred recipe's exact initial
adapters, inputs, physical partitions, checkpoint policy, compiler boundaries,
gradient checks, optimizer and scheduler. The integrated native canary passes
batch sizes 1/2/4/8, finite decoded-weight backward and zero outlier-row error.
Global/cohort eager and compiled repeated losses equal 1.4450643062591553 and
1.2848907709121704 respectively. Both longest-input backwards and the global
nonzero-LR optimizer update pass; that update takes 71.355 s versus 71.272 s
previously. Warm-up and every replay change the FP32 master adapters.

| Measured replay | Preferred recipe seconds/step | Tiled packing seconds/step | New graphs |
| --- | ---: | ---: | ---: |
| 1 | 12.368 | 12.892 | 0 |
| 2 | 12.566 | 13.021 | 0 |
| 3 | 12.503 | 12.759 | 0 |

All thirty measured steps average **12.890800 s**, median **12.568111 s**,
range **9.630355–17.274396 s**, versus preferred **12.479019 s**. This is a
**3.300% regression**, delivering **10,195.9 actual tokens/s** rather than
10,532.3. The predeclared 5% useful-gain criterion fails. Peak allocated memory
is unchanged at **138.638038 GiB**; reserved memory is **143.250 GiB** versus
143.289 GiB. Warm-up takes **196.314 s** with six new graphs; total graphs then
remain at 24. Isolated packing-call gains do not establish lower complete-step
time or meaningful peak-memory savings. No attribution to a particular GPU
instruction, bitwise gradient parity, held-out quality or serving parity is made.

Keep `row_inductor_fused_timing.yaml` as the preferred bounded FP4 configuration.
The new path is opt-in and requires per-token fused scaling, native FP4, BF16
backward and successful packing canaries. Attention, BF16 LoRA computation and
FP32 adapter masters retain their existing scope. Following the blog further
should target its optimized selector and tensor handling rather than assuming
that its attention or backward uses FP4. The blog's TransformerEngine path is a
separate backend to pin and validate before any replacement claim.

The audit `results/fp4_row_inductor_packed_timing/analysis.json` verifies all
twelve source/config hashes against `2ce0f97`; the launcher records parent
`e4c4012` plus those exact hashes. GPU/software, initialization, trainable names,
source jobs, example order, every physical partition, token counts and LR
sequences match the preferred baseline. Every same-batch mean is slower
(0.62–5.97%). Final master hashes differ across replays; deterministic complete
training trajectories are not claimed.

Reports, logs, canaries and both FP32 checkpoints are collected locally and
remain on the persistent network volume. Local checkpoint checksums equal the
remote receipt: cohort `966e98108e69768e8ff387aa867c9f306212bdd3d976f0a4258a3073ab00b8e4`;
global `941e242fe4d306a581aad828862cf898440981c16b63b1c622de7e627c86aa44`.
Thirty-two focused CPU tests, Ruff and diff checks pass. The campaign is complete;
the B200 is idle with no further run queued.

## Shared gate/up activation packing

The first sequential throughput candidate shares each MLP gate projection's
normalized packed FP4 activations, block scales and FP32 row maxima with its up
projection. Matching weak tensor identity and version prevents reuse for changed
inputs; the up projection consumes the entry once. Original higher-precision
LoRA inputs and decoded-BF16 base dX arithmetic remain unchanged. Native output
and input-gradient comparisons are exact across three widths and both ordinary
and non-reentrant checkpointed execution.

`row_inductor_shared_timing.yaml`, source `22332c6`, passes both exact repeated
eager/compiled model gates, longest-input backwards, the global optimizer update
and all warmed replays. Global update: **70.830987 s**. The full cohort records
**22,288 sharing hits**, zero misses and zero pending entries. All 96 native
bases run; no FP4 backward is used.

Thirty measured steps average **12.586694 s**, median **12.208393 s**, range
**9.224992–16.922683 s**, compared with preferred **12.479019 s**. Pass means
are 12.565226/12.707675/12.487181 s. The 0.863% aggregate regression does not
pass the predeclared 5% improvement rule. This is not evidence of a useful
complete-step speedup; retain the original preferred recipe and leave sharing
opt-in. Peak allocated memory is unchanged at **138.638038 GiB**; peak reserved
is 142.468750 GiB. Warm-up takes 147.493574 s with six new graphs; all thirty
measured steps add zero graphs. Warm-up cache differences are excluded from the
speed comparison.

The collected standard-library audit verifies every source hash against
`22332c6`, identical GPU/software, initialization, trainable names, source jobs,
example order, physical partitions, token counts and LR sequence against the
preferred baseline. All gradient/update gates pass. Final replay hashes differ,
so bitwise trajectory parity and held-out quality are not claimed. Forty focused
CPU tests pass; the subsequent selector preparation also passes 43 tests.

Reports, logs and FP32 masters are collected locally under
`results/fp4_row_inductor_shared_timing/` and
`logs/runpod/fp4_row_inductor_shared_timing/`, and remain on the persistent volume.
Local and remote checkpoint SHA-256 match:

- Cohort: `fdb84724ad2fb0db094d2b5e292d3d79b6c37d0fef05cd1c9ba1abcda7f11980`.
- Global: `be5a1b1d4f3f1545b2965f3fc1b78898dcdb3172852919533d63d5e779383c81`.

The next independent candidate uses the optimized FP16 activation selector,
starting from the preferred recipe without activation sharing.

## FP16 activation-selector gate

The independent Triton candidate at `80178f4` uses FP16 candidate products and
FP32 scaled-target MSE, inspired by official TransformerEngine PR3068. Strict
frozen-weight packing is unchanged. Its seven operand cases are deterministic
and finite, with 99.9805–100% logical block-scale agreement and identical
reported aggregate activation squared error. Those aggregate measurements do
not establish identical operands or outputs.

The `(16384, 9216)` outlier case changes 453 block scales and has native-output
relative L2 **0.003665291**, exceeding the predeclared **0.002** limit. The
candidate stops before model loading or optimizer updates. Preserve this
failure; the acceptance limit is unchanged. Passed model-width cases show
1.42–1.95x lower isolated packing-call wall time, which is not a complete-step
speed result. The standard-library audit verifies all fifteen recorded source
hashes against `80178f4`. Reports and logs are collected under
`results/fp4_row_inductor_fp16_selector_timing/` and its Runpod log directory.

The guarded follow-up `row_inductor_fp16_guarded_timing.yaml` tests whether
near-tied choices explain the output mismatch. When the FP16 error difference
is at most `1e-4 * (error4 + error6)`, recompute that entire 16x64 tile using the
pinned strict selector. Zero-error blocks already choose six on both paths.
Floating-point fusion is enabled to match the upstream strict kernel setting;
this also changes the candidate arithmetic contract. The near-tie explanation
is a hypothesis, not an established cause. Require the same seven operand gates
before model loading, followed by the unchanged native/model/gradient gates,
global-longest update and thirty warmed measured steps. Stop at any failed gate;
select only a >=5% complete-step improvement over the 12.479-second baseline.

The guarded candidate at `1ac8d00` completes all gates. Its seven isolated
cases have zero logical scale disagreements and zero native-output difference.
Both repeated eager/compiled model losses equal their preferred baseline
values; longest-input backwards and the global nonzero-LR update pass. The
global update takes **69.934439 s**.

Thirty warmed steps average **12.458150 s**, median 12.271146 s, range
9.193135–16.696765 s. Pass means are 12.624979/12.546135/12.203336 s, all with
zero new graphs. The **0.167%** mean reduction fails the predeclared 5% useful
gain criterion. Peak allocated memory remains exactly **138.638038 GiB**;
reserved is 142.175781 GiB. Warm-up takes 145.955048 s and adds six graphs.
Keep the strict selector in subsequent independent experiments. Faster isolated
packing and repaired canary parity do not establish a useful complete-step
speedup or held-out quality.

The collected audit verifies all fifteen source hashes against `1ac8d00` and
identical GPU/software, initialization, trainable names, source jobs, example
order, physical partitions, token counts, checkpoints and LR sequences against
the preferred baseline. All 96 native bases and decoded-BF16 backwards run,
with zero FP4 backwards. Final replay hashes differ; bitwise training-trajectory
parity is not claimed. The tile-wide strict fallback is part of this recorded
contract, rather than a general proof of elementwise batch independence.

Reports, logs and both FP32 masters are collected locally under
`results/fp4_row_inductor_fp16_guarded_timing/` and its Runpod log directory, and
remain on persistent storage. Local/remote checkpoint SHA-256 match:

- Cohort: `0032a8f424adef7f46fc3ed4e1b4a4e3923c70f581f420e44be68b6dea7e360a`.
- Global: `60e3b008195c9a2193bf99943bd51f9b4713b5ff22af350d3ec42f7c44fb4d18`.

## Optimized Inductor profile

`row_inductor_profile.yaml` at `7616ae7` profiles the preferred strict-selector
FP4 recipe after ten warm-up steps. Native and exact repeated eager/compiled
model gates, both longest-input backwards and the global optimizer update pass.
Global update: 71.335668 s. The single instrumented batch has 188,965 actual tokens,
202,576 padded tokens and thirteen physical batches, matching the earlier
profile's logical workload. Profiling performs no optimizer update and leaves
the warm-up final adapter hash unchanged; it adds no compiler graphs.

Chrome-trace analysis excludes enclosing GPU annotations and counts **126,394
kernel launches**, **15.566277 s** summed kernel activity, **15.555028 s** union
of kernel intervals and 0.075721 s of copies/memsets. Instrumented wall duration
is **22.900770 s**; neither it nor the warm-up times are the thirty-step timing
selection. Kernel activity by enclosing CPU scope is:

| Scope | Kernels | Summed kernel seconds |
| --- | ---: | ---: |
| Native FP4 forward | 11,440 | 1.188083 |
| Decoded-BF16 base backward | 2,080 | 0.643553 |
| FLA | 7,956 | 3.939547 |
| Other | 104,918 | 9.795093 |

The 1,716 activation quantization kernels total 0.560 s. The two leading SDPA
backward kernels each total approximately 1.69 s; substantial other work remains
in casts, adds, multiplies and dense operations. These observations support
testing compiler boundaries and recomputation. Recorded GPU durations include
profiling effects and overlap; they do not predict an additive end-to-end saving.
The profile is of the preferred optimized recipe, replacing the earlier AOT
profile as the current attribution reference.

Collected `profile_audit.json` verifies all seventeen source hashes against
`7616ae7`, identical initialization/trainables, GPU/software, order, actual and
padded tokens, physical batches, checkpoints and LR sequences against the
preferred baseline. Reports, raw trace, logs and FP32 masters are collected
locally and persist under `results/fp4_row_inductor_profile/`. Local and remote
checkpoint checksums match:

- Cohort: `3d4c8f093884d898d682c3a6ec228f8c80558096feffa9cc254b33631c553b39`.
- Global: `a4f3206054ec8af7111fbafc76eee090bdabbc72c35422310f3812c24b56e3fd`.

## Compiler-visible native projections

`row_inductor_visible_timing.yaml` at `7616ae7` replaces Python-disabled frozen
projections with opaque compiler-visible forward/dX operators. Their real
kernels preserve the existing row normalization, CUTLASS output/rescaling and
decoded-BF16 backward. Fake kernels describe output metadata only; per-runtime
CPU tensor keys support graph reuse without GPU `.item()` or projection-specific
integer guards. Keys are process-local and rebuilt on model construction.

All twelve native Inductor cases have exact outputs and input gradients. Both
repeated eager/compiled model gates equal their preferred baseline losses.
Longest-input backwards and the global nonzero-LR update pass; global update
takes **70.407244 s**. All 96 native bases and BF16 backwards run, with zero
FP4 backwards.

Thirty measured steps average **12.137727 s**, median 11.875517 s, range
9.177585–16.107747 s. Replay means are 12.189854/12.092683/12.130643 s, each
adding zero graphs. Every matched batch mean improves (1.81–5.30%); aggregate
step-time reduction is **2.735%**, with **10,828.5 actual tokens/s**. This is a
small measured improvement but falls below the predeclared 5% useful-gain
criterion. Keep it opt-in rather than carrying it into the next independent
memory-policy comparison. Warm-up adds five graphs and takes 142.541145 s;
steady total is 20 graphs rather than 24. Peak allocated memory is **138.789386
GiB** versus 138.638038 GiB; reserved is 142.488281 GiB.

The collected audit verifies all seventeen source hashes against `7616ae7`,
identical GPU/software, initialization/trainables, source jobs, order, token
counts, partitions, checkpoints and LR sequences. Final replay hashes differ;
bitwise complete trajectories, held-out quality and serving parity are not
claimed. The predeclared wider BF16 full-attention compilation follow-up uses
the preferred eager-native bases and first requires a forward-only diagnostic
with the stable RMSNorm/SiLU fences and precision-cast preservation.

Reports, logs, canaries and FP32 masters are collected under
`results/fp4_row_inductor_visible_timing/` and its Runpod log directory, and remain
on persistent storage. Local/remote checkpoint SHA-256 match:

- Cohort: `b554f9c5ac158f9ca4cbb3e4ed61e8fb5cbc07729339af9a9585b4bd0dedb40e`.
- Global: `09a1f42c5ba13feaf2cf35e6cb42dc7730c7e87d342ba52d52e35b7c7e5dd8d5`.

## Wider BF16 attention compilation rejected

`row_inductor_attention_diagnostic.yaml` at `71cbf50` restores
`full_attention_and_linear_shell` compilation, retaining eager RMSNorm/MLP SiLU,
BF16 precision-cast preservation, strict FP4 packing and the preferred
eager-native bases. Fused-row and native kernel preflights pass.

The forward-only model gate fails: repeated eager loss is **1.284890771**, while
both compiled calls produce **1.237992048**, a **-3.650%** change outside the
unchanged `0.01 + 0.01 * abs(eager)` tolerance. Prefixes 0/1/4/8 retain the eager
loss; prefixes 16/32 produce 1.237992048 and prefix 24 produces 1.294600964.
The nonmonotonic prefix pattern does not identify the responsible operation.
Do not infer that BF16 attention arithmetic is invariant under this compilation
policy merely because its storage/compute dtype remains BF16.

The initial FP32 adapter hash is unchanged, with zero optimizer updates. This
is a completed diagnostic with a rejected numerical gate, not a passing training
campaign. The conditional full timing configuration is not launched. Retain
`decoder_shells_without_token_mixers` for the independent memory-policy test.
Collected `analysis.json` verifies all seventeen source/config hashes against
`71cbf50`; reports and logs persist under
`results/fp4_row_inductor_attention_diagnostic/` and its Runpod log directory.

## Reference offload and checkpoint capacity

The memory-policy intervention at `da0ac64` moves the original frozen BF16 MLP
reference parameters to CPU after initial probes. Packed FP4 forward weights
and decoded BF16 backward weights stay on GPU. The 96 bases release exactly
**4,529,848,320 bytes (4.21875 GiB)** of allocated GPU reference weights, matching
the observed allocation decrease. The decoded cache remains necessary for the
selected BF16 dX backward; it is not the original unquantized reference weight.
These GPU caches are rebuilt on model construction, rather than restored as
training state from persistent storage. Dense reference probes temporarily copy
one CPU weight back to the input device outside measured training steps.

`row_inductor_offload_4cp_timing.yaml` reduces checkpointed layers from twelve to
`[0, 10, 21, 29]`, retaining the strict selector, eager token mixers and original
16,384 padded-token physical batching budget. Native, fused-row and exact
offload output/input-gradient/FP32-adapter-gradient canaries pass. Global repeated
eager and compiled losses all equal **1.445064306**. The longest-input backward
then fails with a CUDA OOM: a 460 MiB request at 176.12 GiB allocated, 1.42 GiB
reserved but unallocated, and 39.75 MiB free. No optimizer step runs. This is a
capacity rejection, not a measured timing result or failed arithmetic gate.

Collected `analysis.json` verifies all twenty source/config hashes against
`da0ac64`, the four checkpoint indices, passed canaries, released reference
bytes and empty optimizer-step list. Reports and logs persist under
`results/fp4_row_inductor_offload_4cp_timing/` and its Runpod log directory.
The predeclared eight-checkpoint fallback passes the longest-input backward and
global nonzero-LR update. The update takes **69.068986 s** with **160.521597 GiB**
peak allocated memory. Eager/compiled losses remain exact and its gradient norm
is finite. Final artifact writes then exhaust the persistent volume quota,
confirmed independently by an `OSError: [Errno 122] Disk quota exceeded` write
probe. Its screen remains `running` and campaign status is empty; do not count
it as a complete timing campaign. Preserve reports, logged update and checkpoint
under `results/fp4_row_inductor_offload_8cp_timing/`; the collected analysis
verifies its twenty source hashes against `da0ac64` and labels the interruption.

Clearing the rebuildable uv package cache releases **11.8 GiB** and restores
writes, preserving the installed environment, model/data, compiler/kernel caches
and all experiment artifacts. The unchanged training policy retries in a fresh
directory via `row_inductor_offload_8cp_retry_timing.yaml` at `5f1b273`. The
conditional 24,576-token physical-budget configuration uses the same eight
checkpoint indices. Record completed timing before selecting either recipe.

## Eight-checkpoint warmed timing

`row_inductor_offload_8cp_retry_timing.yaml` at `5f1b273` completes all standalone,
exact repeated eager/compiled model, longest-input backward and update gates.
Its global update takes **69.273487 s**, with **160.521597 GiB** peak allocated.
All 96 FP4 bases run, with 60,960 forward calls and 47,232 decoded-BF16 backward
calls; zero FP4 backwards. Twelve checkpoints in the preferred baseline become
eight, while the strict selector, eager native/token-mixer boundaries and
16,384-token batching policy remain unchanged.

Thirty measured steps average **11.995786 s**, median 11.744045 s, range
9.009857–15.793477 s. Replay means are **12.019676/11.988539/11.979145 s**,
each adding zero graphs. All ten matched batch means improve by 1.77–6.59%,
but aggregate reduction is **3.872%**, below the predeclared 5% useful-gain gate.
Actual throughput is **10,956.6 tokens/s**. Warm-up takes 141.484255 s and adds
six graphs, reaching 24. Peak measured allocated memory rises from 138.638038
to **157.855171 GiB** (reserved 161.875 GiB): the 4.21875 GiB weight release is
spent on saved activations as recomputation decreases. Weight offload is not
equivalent to a lower whole-step memory peak.

The collected audit verifies all twenty source/config hashes against `5f1b273`,
identical initialization/trainables, GPU/software, source jobs, indices, actual
and padded tokens, physical partitions and LR sequences, and explicitly records
the changed checkpoint indices. Final replay hashes differ; complete bitwise
trajectories, held-out quality and serving parity are not claimed. Do not adopt
the checkpoint-only intervention under the frozen selection rule. Its separate
24,576-token physical-budget follow-up retains the eight-checkpoint layout and
requires the same numerical gates and full replay protocol.

Reports, logs, canaries and FP32 masters persist under
`results/fp4_row_inductor_offload_8cp_retry_timing/` and are collected locally.
Local/remote checkpoint SHA-256 match:

- Cohort: `0f33213e0c79dc0b49486f881b16508ca7c3487c448a108154cd13cf9eccc256`.
- Global: `1dbddcb43aba5fa2316af6c80f83abe64593597296f0f35bf61e166b4d220298`.

## Larger physical batches rejected

`row_inductor_offload_8cp_24k_timing.yaml` at `5f1b273` retains the eight
checkpoints and CPU reference weights, raising only the physical padded-token
budget from 16,384 to 24,576. All standalone, exact repeated eager/compiled,
longest-input backward and update gates pass. The global update takes
**69.038855 s**, with **160.521597 GiB** peak allocated memory. The cohort has
the same initialization, 320 examples, order, 1,314,331 actual tokens per replay,
effective batch 32, maximum physical batch eight and LR sequence.

Physical batches decrease from **115 to 102 per replay**, while padded tokens
increase from **1,411,298 to 1,447,858 (2.591%)**. Thirty measured steps average
**13.745459 s**, median 13.328680 s, range 9.797941–18.809362 s. Replay means
are **13.740951/13.746868/13.748559 s**, each with zero new graphs. This is
**10.149% slower** than the preferred 12.479019-second FP4 baseline and **14.586%
slower** than the eight-checkpoint smaller-budget run. Throughput is **9,561.9
actual tokens/s**. The changed partitions do not improve complete-step time;
counts and padding alone do not attribute the slowdown to a particular kernel.
Do not adopt the larger budget.

Warm-up takes **221.226338 s** and adds six graphs, reaching 24; its first
69.22-second step includes compilation and is excluded from timing. Peak measured
allocated memory is **157.855171 GiB**, reserved 161.853516 GiB. All 96 native
bases run, with 54,720 forwards, 42,240 decoded-BF16 backwards and zero FP4
backwards. The original reference offload still releases exactly 4.21875 GiB.

Collected `analysis.json` verifies all twenty source/config hashes against
`5f1b273`, initialization/trainables, GPU/software, source jobs, actual tokens,
order and LR, and records the changed checkpoint and physical-batch policies.
Partitions and padding are identical across this campaign's four replays;
local/remote FP32 master checksums match. Final replay hashes differ. Reports,
logs, canaries and masters are collected locally and persist under
`results/fp4_row_inductor_offload_8cp_24k_timing/`. Checkpoint SHA-256:

- Cohort: `8f16721501c7bd7556bb986b64a6fb63e80988c1f1c0a5c1e0a6a42fa86a044d`.
- Global: `1ac23556bd413918cd7f8cf38317cef151de40294d94a08423c17a18d70a98c9`.

## Sequential campaign selection

All four proposed interventions and their predeclared follow-ups are complete.
Against the same 12.479019-second FP4 baseline, shared packing is 0.863% slower,
the guarded FP16 selector is 0.167% faster, compiler-visible native projections
are 2.735% faster, and reference offload plus eight checkpoints is 3.872% faster.
Four checkpoints exceed capacity; wider attention compilation fails its forward
numerical gate; larger physical batches are 10.149% slower. None meets the frozen
5% useful-gain criterion, so retain `row_inductor_fused_timing.yaml` as the
preferred bounded recipe. Keep passing alternatives opt-in and preserve negative
results. Their individual gains do not establish the performance of a combined
recipe, a pure-BF16 comparison, held-out training quality or serving parity.
The B200 is idle after artifact collection; no training run remains queued.

## Detailed optimized-profile attribution

`experiments/fp4_stability/profile_breakdown.py` reanalyzes the already collected
optimized Inductor trace. No new GPU experiment or optimizer update is launched.
The trace SHA-256 is
`50f8629950ea7075bad4270a727ef17bc007ff6fee1a1365875f742f0e100441`.
All 126,394 GPU kernels map to CPU external IDs. Calls and summed durations
reconcile with each original coarse scope; CPU/annotation durations are never
added to GPU durations. The 9.795093-second `other` bucket partitions as follows:

| Observable operation family | Kernel calls | Summed GPU seconds |
| --- | ---: | ---: |
| Full-attention backward | 232 | 3.397447 |
| Eager pointwise operations | 41,576 | 1.756229 |
| Casts and tensor copies | 15,684 | 1.180349 |
| Other matrix multiplication | 17,628 | 1.117802 |
| Compiled pointwise operations | 13,832 | 0.806079 |
| Full-attention forward | 104 | 0.767809 |
| Causal Conv1d | 780 | 0.221179 |
| Layout/indexing | 1,117 | 0.182789 |
| Reductions | 3,660 | 0.180699 |
| Normalization | 780 | 0.123195 |
| Weight dequantization | 9,464 | 0.060910 |
| Remaining named kernels | 61 | 0.000606 |

Full attention accounts for **4.165257 s**, independently of FLA's 3.939547 s.
Flash SDPA forward runs 64 times (0.532175 s), while memory-efficient SDPA
forward runs 40 times (0.235634 s). Their backward CPU operators launch kernels
totalling 1.706662 and 1.690785 s respectively. Mixed dispatch is observed;
tensor shapes and mask arguments were not recorded, so the dispatch cause is
not established. The older FA4 negative timing was a different fixed-singleton
NF4/checkpoint workload. It does not settle performance on this padded native
FP4 cohort. Any new comparison must preserve mask semantics, eager attention
boundaries and the numerical/backward/update gates.

The largest generic CPU-operator attributions are `aten::copy_` (1.180349 s),
`aten::mm` (1.117733 s), `aten::mul` (0.885840 s), and `aten::add_` (0.417662 s).
Pointwise operations total **2.562307 s** across **55,408 kernels**. Names do not
identify model modules: the existing trace has no input shapes, Python stacks
or module annotations. Do not call all these kernels LoRA, optimizer work or
quantization. In particular, this profile performs no optimizer update.

FLA phase attribution uses a forward CPU scope nested inside an autograd
backward scope to identify outer checkpoint replay:

| FLA phase | Kernel calls | Summed GPU seconds |
| --- | ---: | ---: |
| Original forward | 2,808 | 0.961489 |
| Outer checkpoint forward replay | 1,404 | 0.480863 |
| Backward | 3,744 | 2.497194 |

The FLA backward additionally reconstructs intermediates internally:
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` takes **0.374450 s** and
`recompute_w_u_fwd_kernel` takes **0.261197 s**, both already included in the
2.497194-second backward total. Installed FLA 0.5.2 explicitly recomputes them;
its autograd context does not retain h/w/u. Other large backward kernels are
`prepare_wy_repr_bwd_kernel` (0.782174 s), `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
(0.478951 s), and `chunk_bwd_kernel_dqkwg` (0.401398 s). Selective retention of
expensive FLA intermediates is a distinct hypothesis from removing entire layer
checkpoints; its memory and correctness are untested. Do not assume the previous
four-checkpoint OOM permits it, or add internal and outer replay times twice.

Native FP4 forward's 1.188083 s includes **0.559620 s** activation packing,
**0.197589 s** row normalization, **0.127973 s** output rescaling and **0.256116 s**
native GEMM, with small remaining operations. Each main component runs 1,716
times. Preparing/rescaling operands costs substantially more profiled activity
than the FP4 GEMM. The preceding tiled packing and guarded selector tests did
not yield useful whole-step gains. A different fused epilogue or packing design
needs shape-specific measurements and must preserve the established BF16
rounding boundaries; merely reducing launches does not establish improvement.

An upstream candidate is [Qwen FlashQLA](https://github.com/QwenLM/FlashQLA),
version 0.1.3 at `da06429d54b0f577de0a638f451ac8f0b395e0ac` (2026-09-30).
Its [entrypoint source](https://github.com/QwenLM/FlashQLA/blob/da06429d54b0f577de0a638f451ac8f0b395e0ac/flash_qla/ops/gated_delta_rule/chunk/__init__.py)
includes SM100 forward/backward, q/k normalization, variable-length sequences
and grouped q/k head handling. The library combines operator fusion with
optional intra-card sequence partitioning. Its published comparisons use FLA
0.5.0 and different kernel/software workloads, so they are not a speed estimate
against our FLA 0.5.2 recipe. Benchmark the exact 128-dimensional, 32-value-head
Qwen4B inputs, outputs and all gradients before a model comparison; test ordinary
execution and automatic partitioning separately. It has not been installed or
benchmarked here, and no dependency/default is changed.

The resulting priority is to test an optimized Gated DeltaNet backend and
investigate full-attention dispatch, then isolate pointwise/cast module origins
with a more annotated profile before changing fusion/checkpoint boundaries.
Quantization remains a smaller measured target. Store the detailed receipt as
`results/fp4_row_inductor_profile/profile_breakdown.json`. This is one instrumented
batch with profiling overhead, not a new ordinary-step timing or quality result.

## FlashQLA kernel screen and input-dtype boundary

The pinned experimental FlashQLA install uses source
`da06429d54b0f577de0a638f451ac8f0b395e0ac` (0.1.3+da06429), TileLang 0.1.12
and TVM-FFI 0.1.11, in `.cache/kernels/flashqla-da06429`. The main locked
environment, FLA 0.5.2, Triton 3.7.1, Conv1d and native FP4 package are unchanged.
Source archives, installed Python/CSV checksums and compiler caches persist on
the network volume. This screen uses the same B200; automatic intra-card
partitioning is disabled in the following results.

The initial BF16-only isolated screen passes all seven shapes, output relative
L2 <= 0.01 and every q/k/v/g/beta gradient relative L2 <= 0.02. Actual maxima
are 0.004388 output and 0.005765 gradient. Ten warm-up calls and thirty measured
calls per backend/phase exclude first compilation. Forward+backward latency
falls 28.96--44.68% against BF16-input FLA. The receipt is
`results/fp4_flashqla_canary/canary.json`, SHA-256
`ecd104c10842e51323c581efa712ad875171b95b8fd4dde6960e33293479186c`.
Frozen helper/entrypoint snapshots reconcile with the receipt source hashes.

The first whole-model attempt, `row_flashqla_timing.yaml` at `7839961`, stops
before any optimizer update: FlashQLA rejects q's input dtype with
`FlashQLA only supports bfloat16 and float16`. The reference model passes its
original initial adapter identity and native arithmetic gates; the new backend
never reaches its compile/global-update gates. Preserve the failed run under
`results/fp4_flashqla_timing/`. BF16 projection compute does not guarantee BF16
GDN interface operands. Do not use the BF16-only kernel timing as a matched
estimate of this original model path.

The explicit follow-up at `4a60ec0` casts q/k/v/beta to BF16, retains FP32 g,
and returns o in the original q dtype; casts remain differentiable to FP32
master adapters. The FP32-input isolated screen compares original FLA, FLA
with the identical boundary, and FlashQLA with that boundary, including casts
and allocations in the synchronized latency. Both candidates must pass the
same original-FLA output and all-gradient gates before whole-model use.
All seven shapes pass; FlashQLA maximum output relative L2 is 0.005205 and
maximum individual gradient relative L2 0.006000.

| Physical shape | Original FLA, ms | FLA + BF16 boundary, ms | FlashQLA + boundary, ms |
| --- | ---: | ---: | ---: |
| 1 x 257 | 7.127 | 7.615 | 5.793 |
| 1 x 4,096 | 7.253 | 8.137 | 5.475 |
| 1 x 16,384 | 14.172 | 10.108 | 7.240 |
| 1 x 29,696 | 24.156 | 16.895 | 9.615 |
| 2 x 8,192 | 13.954 | 8.575 | 5.783 |
| 4 x 4,096 | 14.023 | 8.297 | 5.521 |
| 8 x 2,048 | 13.472 | 8.354 | 5.664 |

These are isolated forward+backward means, not complete optimizer-step times.
FlashQLA reduces latency 18.72--60.63% against the original FP32-input FLA and
23.93--43.09% against FLA with matched casts. Casting alone slows the two short
singletons but helps the longer/multi-example shapes. Synthetic zeroed tails
are included for padded shapes; no raw model activations are collected.
Receipt: `results/fp4_flashqla_fp32_canary/canary.json`, SHA-256
`813d86ecbb2ff0b67df84624573f0ec9ed3a289d89d7f026667da3ee366b81eb`.
Helper and entrypoint snapshots match its source hashes.

`row_flashqla_bf16_boundary_timing.yaml` is the separate whole-model follow-up.
It retains the original strict FP4 arithmetic, twelve checkpoints, batching,
initialization and ten-warm/thirty-measured protocol. Before updates, record
actual GDN input dtypes and require native-model loss agreement plus aggregate
unclipped FP32 master-gradient relative L2 <= 0.05 against original FLA.
The model run fails before any optimizer update. Actual q/k/v/g/beta inputs
are all FP32. All 256 master gradients are finite, but aggregate relative L2 is
**1.009731**, versus the 0.05 gate. Forward losses also fail: one singleton
changes **1.179026 -> 0.828410** and the padded canary changes
**1.445064 -> 1.477198**. Large errors affect early MLP and full-attention LoRA
B gradients, not only the replaced GDN modules. Both arithmetic preflights and
the initial master identity pass. Twenty-one source/config hashes reconcile
with `4a60ec0`. The collected failed receipt is
`results/fp4_flashqla_bf16_boundary_timing/`; it has zero optimizer steps and
no accepted whole-step timing. Cancel the predeclared training controls and
automatic-partitioning follow-ups rather than training past these gates.

`row_flashqla_boundary_diagnostic.yaml` adds a bounded failure analysis with no
updates: compare the same BF16 boundary using FLA on the same losses and
reference gradients, then compare each linear layer's candidate outputs on
three probe batches. Shadow calls return original FLA outputs so later-layer
inputs stay matched. The diagnostic at `a858ddc` finishes at the expected failed
gate, with zero updates and no diagnostic errors. FLA with the same boundary
also fails: aggregate gradient relative L2 is **1.184213**, compared with
**1.009446** for FlashQLA on this replay. All gradients are present and finite.
Original-FLA and FlashQLA forward losses repeat exactly across the two model
attempts; their gradient error totals are close but not bitwise identical.

The 72 shadow comparisons cover all 24 linear layers on three probe batches,
with original-FLA outputs returned through the model:

| Candidate | Mean local output relative L2 | Maximum local output relative L2 |
| --- | ---: | ---: |
| FlashQLA + boundary | 0.003198 | 0.007502 |
| FLA + boundary | 0.003301 | 0.007484 |

All local outputs are finite. Small local changes coexist with large end-to-end
loss/gradient differences. The precision boundary alone is already problematic
for this hybrid FP4 recipe; this evidence does not isolate MLP quantization,
normalization, an individual operand, or their interactions as the amplifier.
Do not label FlashQLA's backward implementation broken from these results, or
claim the isolated kernel gain is a stable training-speed improvement.

The receipt is `results/fp4_flashqla_boundary_diagnostic/fouroversix/screen.json`,
SHA-256 `2951d8734430d668fd002df0c93a453a880451a318a0e2a2cd85e6feb2650be8`.
It omits the launch-time Git revision, but all 21 source/config hashes verify
against `a858ddc`; the separate analysis records that verified revision. All
receipts, logs and source snapshots are collected and remain persistently saved.
The B200 is healthy and idle; no timing/control/automatic-partitioning run is
queued. Retain the preferred strict FP4/FLA configuration and the original
NF4 default. A future candidate needs a more faithful precision boundary plus
the same actual-model gates before optimizer work or timing.

A future NF4-vs-FP4 comparison must give both recipes the selected attention
backend; this backend can affect both precision recipes. Validation: 46 focused
CPU tests, Ruff and shell syntax checks pass.

## NF4/BF16 FlashQLA integration

The user requested NF4 storage/BF16 compute as the first integration target.
The original NF4 projections still send FP32 q/k/v/g/beta to GDN; this is not
specific to the native FP4 MLP replacement. The initial no-update model
diagnostic uses the selected NF4 compile policy, twelve checkpoints, adaptive
batching, and the same frozen FP32 initial masters as the FP4 screen.

The plain BF16 q/k/v/beta boundary fails on NF4 as well. Adapter-gradient
relative L2 is **0.667101** for FlashQLA and **0.489265** for FLA with
identical casts, against the unchanged 0.05 gate. Maximum forward-loss gap is
0.052118 absolute / 3.53% relative; all 256 gradients are finite. The 72
matched-input shadow calls have mean output error 0.003181 for FlashQLA and
0.003265 for cast FLA. Zero optimizer updates occur. This failure is smaller
than the FP4 case but still unacceptable under the predeclared gate.

Receipt: `results/nf4_flashqla_boundary_diagnostic/nf4/screen.json`, SHA-256
`85c3923d0c3dc3490871edb9afa84ec888247618b81b542ea48665420efb1a2c`.
All 21 source/config hashes reconcile with `d74c742`, although the launch
Git field names the preceding implementation commit `9cd075f`. The derived
analysis records the verified revision; logs and receipts are collected.

The follow-up preserves beta/g in FP32 and uses the pinned FLA FP32 Q/K L2
normalization before casting only the kernel operands; disable in-kernel
normalization so it executes once. This precision boundary is differentiable
and leaves FP32 master adapters intact. The first policy retains BF16 GDN
q/k/v; a predeclared FP16-core fallback offers more mantissa precision while
retaining NF4 storage and BF16 projection compute. It must be labeled as such.

The revised BF16 isolated canary passes all seven shapes, maximum output
relative L2 **0.004654** and maximum individual operand-gradient error
**0.005178**. Its matching FLA-cast control also passes. Receipt:
`results/nf4_flashqla_bf16_precise_canary/canary.json`, SHA-256
`6def4ca42735636c92676beb5d959cf52074cc515e161dc69b761f2e21718465`. Frozen helper and entrypoint snapshots match the receipt.
The revised BF16 full-model diagnostic still fails before updates: FlashQLA
gradient relative L2 **0.448167**, cast FLA **0.383393**. The largest
absolute loss gap remains 0.052118; preserving gates/normalization reduces
the gradient discrepancy but is insufficient. All gradients remain finite.
The receipt `results/nf4_flashqla_bf16_precise_diagnostic/nf4/screen.json`
has SHA-256 `b44d9191c31232b9c32c718c1798ee0ce9b36c4cfe88779435eb027a71247da4`.
All 21 source/config hashes verify against `72b19ec`. Continue only the
predeclared FP16-core isolated/model diagnostic; BF16 timings are canceled.

The first FP16 isolated attempt fails before numerical checks because
TileLang emits `cutlass::half_t` into a masked 256-bit output store whose
`pack_float16x4` helper accepts CUDA `half`. Sixteen C++ conversion errors
occur at the same generated line. Preserve its receipt/log under
`results/nf4_flashqla_fp16_precise_canary/`.

The scoped fix at `ce389fd` adds an overload with the identical bit-packing
body for `cutlass::half_t`; it retains the original overload and does not
change FlashQLA math. Only the isolated TileLang target is patched.
Original header SHA-256 is
`da858d5cf8a7f5f780aced7a914059135d3ef9011dc9198d51d3d52624118f4f`,
patched header SHA-256
`a63307562f4c1d8b191a8da0f29bc2de19523a0bde42c67993a56b32396849f9`.
The script rejects unknown originals, verifies idempotence and drift, and
records both hashes plus its own hash in the install manifest. Backend
loading verifies those identities; the locked main environment is intact.

The patched FP16 canary, using fresh `fp16-pack-v1` compiler caches, passes
all seven shapes and every operand-gradient gate. Maximum output relative
L2 is **0.001603**, maximum individual gradient error **0.003192**. Its
matched FLA precision-boundary control also passes. Receipt:
`results/nf4_flashqla_fp16_precise_patched_canary/canary.json`, SHA-256
`b25cd727bdc1ffc23fab73e8442dddc07f30772b2d8ae3b10ea2eb2a16187369`.
Helper, entrypoint, patch-script and header snapshots all verify.
The FP16-core NF4 whole-model diagnostic also fails before updates:
gradient relative L2 **0.400802**, versus **0.387613** for cast FLA.
All gradients are finite; several singleton loss gaps still exceed the
original tolerance. Receipt: `results/nf4_flashqla_fp16_precise_diagnostic/nf4/screen.json`, SHA-256
`9e7acd021290b1e6d0574822d4849885df7bacd998ff7f30728f198cb011f963`.
All 21 source/config hashes verify against `ce389fd`. No full-replacement
optimizer updates or accepted whole-model timings occur in any variant.
Validation: 51 focused CPU tests for the precision/compiler changes.

A separate strict partial-layer diagnostic now tests BF16 kernel operands
with FP32 gates/normalization only in the final 12, 8, 4, 2, then 1
linear-attention layers, stopping at the first passing original-FLA
loss/gradient gate. Earlier layers retain original FLA and FP32 operands.
This addresses error propagation without relaxing thresholds; it must be
reported as partial FlashQLA. No timing run starts unless its selected
subset passes. Layer selection/restoration and bounded configuration checks
bring focused CPU validation to 61 passing tests.

The BF16 partial-layer screen finishes at the expected failed gate, with
zero updates. All subsets fail at least one forward loss; aggregate
gradient relative L2 falls sharply for late-layer-only replacement:

| Final linear-attention layers replaced | Gradient relative L2 | All gates pass |
| --- | ---: | --- |
| 12 | 0.079262 | No |
| 8 | 0.016508 | No |
| 4 | 0.016642 | No |
| 2 | 0.014503 | No |
| 1 | 0.010786 | No |

Receipt `results/nf4_flashqla_bf16_layer_diagnostic/nf4/screen.json` has
SHA-256 `9a1f20c6384763bee9aad6b4266b1efc7348442f6ec0b6c980d045fea1a66d9b`.
All 21 source/config hashes verify against `5326e12`; launch revision
names the preceding `ce389fd`. Preserve every case; no subset is accepted.

Forward losses change in coarse increments (roughly 0.025 or 0.051 on
these probes). Source inspection finds the screen's outer BF16 autocast
also includes the final LM-head projection; casting its result to FP32
for the loss does not undo prior rounding. The next separate arithmetic
intervention explicitly runs the frozen head outside autocast in FP32,
leaving decoder precision, head weights, loss definition, data and FP32
LoRA masters unchanged. The matching FLA reference uses the same FP32 head.
The wrapper records the actual original output dtype on its first call.
This is motivated by the existing MIL projection precision finding; do not
claim causality or old-recipe quality parity before matched evidence.

Test uniform FlashQLA first (all 24 linear layers), then the predeclared
late-layer subsets only if needed. Retain all original loss/gradient
thresholds, and give both timing arms the FP32 projection if any case
passes. Validation: 63 focused CPU tests pass.

The FP32-head/partial-layer diagnostic was interrupted at the user's request
before any optimizer updates. Its final-eight-layer canary passed with gradient
relative L2 0.016405, but the changed head/subset has no measured training result
and is not promoted. Preserve its interrupted receipt under
`results/nf4_flashqla_fp32_head_diagnostic/`.

The user narrowed the experiment to exactly ten uniform FlashQLA optimizer
calls and a matched current-recipe FLA control. The new configs retain the
current BF16 head and frozen initial adapters/cohort. They explicitly allow
finite uniform FlashQLA training despite the recorded strict parity failure;
this exception is confined to the ten-step learning diagnostic. All ten ordered
batches warm compilation with backward only and unchanged adapters, followed by
ten optimizer calls (nine nonzero learning rates). Common original-FLA probe
loss is recorded outside synchronized step timing. No head changes, subset
sweeps, separate preflight updates, timing replays or held-out promotion.
Focused CPU validation: 70 passed; six combinations skipped because the bounded
comparison excludes partial layers/profiling. Both configs are frozen before
launch; collect loss, timing, memory and compilation receipts before concluding.

The first uniform ten-step launch stopped with zero updates at the compiled/eager
canary: eager 1.358968 versus compiled 1.327407, both repeatable and finite. Save
this attempt as `results/nf4_flashqla_ten_step_comparison_compile_gate_attempt/`.
To honor the explicitly requested bounded learning comparison, also record this
canary's failed strict gate while permitting finite disagreement in that mode
only. Common probes still evaluate both trajectories with original eager FLA.
The unchanged recipe restarts without a head change or another diagnostic sweep.
The 70 focused tests pass, including exactly ten AdamW calls, no optimizer calls
in warmup, and explicit bounded acceptance of a failed finite compile canary.


### Completed uniform FlashQLA ten-step learning/timing comparison

Both arms complete exactly ten optimizer calls (nine with nonzero LR), with no
updates in the memory preflight or ten ordered warmup backward passes. Match all
320 examples, 1,314,331 actual tokens, shuffled indices, physical partitions,
padded tokens, initial master hash, LR schedule, rank-128 adapters, 12-layer
checkpointing, current BF16 head and compiler policy. All 22 source/config
hashes per arm verify against `992c4f8`; FlashQLA's launch revision is unset,
so the source hashes establish its identity. Both initial common probes equal
**1.2467260063**. Compare common loss with original eager FLA on eight fixed,
truncated training-cohort examples; probe time is excluded from step timing.

| Metric | Current NF4/BF16 + FLA | NF4/BF16 + uniform FlashQLA |
| --- | ---: | ---: |
| Mean synchronized step time (s) | 10.420161 | 9.066682 |
| Median step time (s) | 9.848146 | 8.721489 |
| Step time range (s) | 8.002–13.839 | 7.014–11.622 |
| Actual tokens/s | 12613.3 | 14496.3 |
| Mean training-batch loss | 0.688320618 | 0.688021956 |
| Final common probe loss | 0.926934831 | 0.912150823 |
| Measured peak allocated memory (GiB) | 130.776 | 132.091 |
| New Dynamo graphs during measured steps | 0 | 0 |
| Backward-only warmup time (s) | 113.454 | 114.279 |
| Entire stage wall time, including imports/load/canaries/warmup/export (s) | 596.834 | 606.878 |

FlashQLA reduces measured step time by **12.989%** (1.1493× throughput), and is
faster on every paired batch. Mean training loss differs by only -0.000299.
Its final common loss is 0.014784 lower (about 1.60%), an observation on the
small training probe rather than a quality advantage. Both trajectories remain
finite and show short-run learning. Gradient parity disagreement alone does not
predict catastrophic failure over these ten steps. This supports further
matched validation of uniform FlashQLA; it does not establish long-run stability,
held-out performance, numerical equivalence or promotion of the default recipe.

All 24 GDN layers use FlashQLA with BF16 Q/K/V, FP32 gates and external FP32 Q/K
normalization; all eight softmax-attention layers retain SDPA. No FP16 GDN, FP32
head override or partial-layer rollout. FlashQLA's strict whole-model gradient
gate remains false (relative L2 0.195302), and its compiled/eager loss gate remains
false (1.327407 versus 1.358968). Both finite failures were explicitly permitted
only for this bounded learning diagnostic and remain in the receipt. The FLA
compile canary passes exactly. Both arms emit one Dynamo recompilation-budget
warning during setup under the existing policy; no new graphs appear in the
measured pass, and this timing describes the resulting recipe, including its
fallbacks. No compiler-policy repair or extra timing replay is part of this run.

| Step | FLA time (s) | FlashQLA time (s) | FLA common loss | FlashQLA common loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9.653 | 8.700 | 1.246726 | 1.246726 |
| 2 | 11.590 | 9.919 | 1.199941 | 1.199658 |
| 3 | 9.314 | 7.855 | 1.228531 | 1.224761 |
| 4 | 9.198 | 7.587 | 1.091919 | 1.074316 |
| 5 | 13.839 | 11.622 | 0.981849 | 0.970595 |
| 6 | 10.044 | 8.743 | 0.949739 | 0.933475 |
| 7 | 8.761 | 7.702 | 0.935647 | 0.936196 |
| 8 | 12.283 | 11.534 | 0.946867 | 0.928434 |
| 9 | 11.519 | 9.991 | 0.921861 | 0.917202 |
| 10 | 8.002 | 7.014 | 0.926935 | 0.912151 |

Receipts:
- `results/nf4_fla_ten_step_comparison/nf4/screen.json`, SHA-256
  `7f0cc60308abf80608a43c87093932c748644f5553ab7ae1749724e04f5c04a9`.
- `results/nf4_flashqla_ten_step_comparison/nf4/screen.json`, SHA-256
  `2cb20e98e0899838dab3a48d387401ce97b437dd38fd0b063b83992d9fd73ec7`.
- `results/nf4_flashqla_ten_step_comparison/comparison.json` holds the paired
  loss/timing rows and source/master/partition/schedule checks.

Both saved `fp32_master.pt` artifacts are checked after collection: 256 FP32
adapter tensors, 169,869,312 elements, named order and final tensor digest match
their receipts. Results and logs are collected locally and persist on the pod's
network volume. No further training is launched; the B200 is idle. Validation
remains 70 focused tests passed, six unsupported combinations skipped, Ruff
passed. The failed zero-update compile-gate attempt is preserved separately.


After the ten-step loss/timing explanation, the user explicitly selected uniform
FlashQLA as the default B200 NF4/BF16 LoRA recipe. The decision is recorded in
`docs/decisions/b200_flashqla_training_recipe.md`. This supersedes the earlier
statement that no default was promoted, through user selection rather than a
new held-out quality result. Ordinary training records finite acceptance separately
from failed strict parity, and preserves historical diagnostic contracts.
