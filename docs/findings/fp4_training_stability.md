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
Attention retains NF4 storage and BF16 compute. All initial master hashes match
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
