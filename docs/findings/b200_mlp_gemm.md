# B200 MLP / GEMM screen

Date: 2026-10-05. The user authorized a new single B200 in US-NC-2 and asked
for large MLP / GEMM optimization, with FP4 as the eventual target. The first
BF16 candidates do not meet the predeclared throughput threshold. Keep the
selected BF16 FA4 recipe; no full-model training was launched for these candidates.

## Matched BF16 MLP probes

Use actual Qwen3.5-4B geometry (hidden 2560, intermediate 9216), seeded synthetic
frozen BF16 weights and nonzero rank-128/alpha-256 LoRA adapters with FP32 masters.
The caller uses BF16 autocast; master precision does not imply FP32 matmul.
Six warmups and ten alternating synchronized samples include the complete MLP
forward/backward, all six adapter gradients, input gradient and adapter casts.
Forward-only samples are no-grad inference and cannot be read as training forward.
No optimizer update or checkpoint-quality assessment occurs in these probes.

The default already compiles decoder shells including MLPs. Compare proposed
compiled merging and cuDNN fusion against compiled PEFT within each process;
do not treat ordinary eager-to-compiled speed as a new recipe improvement.
Absolute times across independent processes vary; paired legs are the evidence.

| Complete MLP forward/backward | Tokens | Paired compiled PEFT, ms | Candidate, ms | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Compiled merged gate/up | 4096 | 5.08068 | 4.90509 | 3.46% |
| Compiled merged gate/up | 16384 | 6.14400 | 6.33101 | -3.04% |
| cuDNN LoRA-aware forward graph | 4096 | 4.61873 | 5.02920 | -8.89% |
| cuDNN LoRA-aware forward graph | 16384 | 6.09065 | 7.42512 | -21.91% |

Both candidates preserve outputs and all six adapter gradients exactly in these
samples. Merged input-gradient relative L2 is about 0.00373 and cuDNN about
0.00361; all gradients are finite and independent-row output differences are zero.
They pass the local 1% arithmetic gate but fail the 5% speed gate at both long
shapes. The merged frozen gate/up copy also costs 90 MiB per MLP (2.8125 GiB
for 32 layers), in addition to the original frozen weights. The cuDNN prototype
uses ordinary BF16 backward and uncompiled adapter operations; these results
measure the whole prototype, not isolated GPU GEMM efficiency or kernel count.

Ordinary compiled PEFT versus eager PEFT reduces the 16384-token mean by 16.63%
(7.30255 to 6.08791 ms). The 4096-token compiled leg retains one 13.0378 ms outlier
among ten samples; its mean is 22.89% slower. No samples were removed. Eager
merged gate/up changes the 4096 mean by +1.96% and the 16384 mean by -9.16%.
These controls support neither a new training speedup nor a bottleneck attribution.

The [contract and entrypoints](../../experiments/b200_mlp_gemm/README.md)
record the held-out selection rule, stop condition and possible twenty-update
continuation. The checksum-bound compiled-merge selection was declined before
training. The historical FA4 control remains 4.08648 seconds/update, with its
existing packing and explicit gradient-acceptance records; it was not rerun.

## Receipts and runtime

Artifacts and executed source archives are under
`results/b200_mlp_gemm/pilot02/`, logs under
`logs/runpod/b200_mlp_gemm/`. Probe SHA-256:

- `merged_compiled`: `2f0f717d95d1741b3e899faf8757c65871f8065d62a31f22413890205ab2efa1`.
- `cudnn`: `b1b91867ee3f2e7314fa269702e7c35ba1a7c0794b5777841da20a24146e4fca`.
- `compiled`: `0c8243e8a51591ba034161a165c1299a17d1c3316e57caff1b7f2eb7d169fa79`.
- `merged`: `96941ae494414a43878604e6fec79036bc17c1e83338bbf77e56f44f7a5a5e31`.

The first attempt `pilot01` failed before GPU execution because the fresh
container did not inherit the retained Hugging Face cache environment; its failure
is preserved. The launcher now binds the network-volume cache explicitly.
Runtime: B200 183359 MiB, driver 595.91.07, Torch 2.11.0+cu130, Triton 3.7.1,
Transformers 5.14.1, PEFT 0.19.1, cuDNN Frontend 1.31.0/runtime 9.26.0.51.
NVIDIA source revision: `51d9d06b574222378a3d806009accab098e73705`.
Compiler caches reuse the retained network volume, including shared gpu-0
TorchInductor/Triton and shared cuDNN/CuTe DSL namespaces; no cold-cache test.
Compilation uses 16 workers within the measured 20.4-CPU quota. Input/compiler
imports from the network volume contribute minutes of startup and are excluded
from warmed timings. The authorized pod remains running; lifecycle details are
in [infrastructure](../infrastructure.md).

## Native FP4 forward and input-gradient feasibility

The first native cuDNN FROST NVFP4 probe completes all twelve shape/path cases:
merged gate/up forward, down forward, and their two input-gradient GEMMs, each
at 193, 4096 and 16384 tokens. Unlike the previous Four Over Six training screen,
these input-gradient matrix products use FP4 operands and native FP4 MMA too;
this does not yet integrate the full LoRA MLP or optimizer update.

Frozen synthetic weights use global FP32 scaling and E4M3 scales shared over
16x16 tiles. Both orientations are quantized from original BF16; their decoded
transpose disagreement is exactly zero. Activations/gradient operands use global
dynamic amax and per-row 16-element scales. Gradient codes use RNE, not stochastic
rounding. Output is BF16, with explicit global descaling. This borrows the
transpose-consistent weight scaling documented by
[NVIDIA Transformer Engine](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html),
without claiming its complete training recipe. The native primitive is the
pinned [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) projection/FROST
block-scaled GEMM, using 128x256 tiles on SM100 and the current PyTorch stream.

Native-versus-decoded-operand relative L2 is 0.002777–0.002830 across all cases,
below the predeclared 1% implementation limit; all outputs are finite.
Original-BF16 GEMM disagreement is 0.146105–0.146395, and decoded weight
quantization error is about 0.1113. These are synthetic GEMM errors, not model
loss, adapter-gradient or quality measurements. The FA4-specific user acceptance
does not relax FP4 tolerances or establish quantized-model equivalence.

| 16384-token isolated path | BF16, ms | FP4 prepacked, ms | FP4 including packing/scaling, ms |
| --- | ---: | ---: | ---: |
| Gate/up forward | 1.00427 | 0.73869 | 1.24867 |
| Gate/up input gradient | 1.04818 | 0.56158 | 1.18044 |
| Down forward | 0.56237 | 0.47488 | 0.98844 |
| Down input gradient | 0.52813 | 0.51065 | 0.98221 |

Prepacked gate/up forward/input-gradient time falls 26.45%/46.42%, including
native dispatch and descaling. Dynamic conversion erases this gain: complete
paths are 24.34%, 12.62%, 75.76% and 85.98% slower, respectively. All 4096-token
paths are slower too. Paired samples retain all observations; no cold cache or
whole-training comparison is implied. Weight preparation occurs once and both
packed orientations occupy 79,626,256 bytes per synthetic MLP; original BF16
weights and diagnostic decoded copies remain resident. The initial receipt's
`pack_seconds_including_first_jit` includes diagnostic decoder dispatch after
the packing synchronization and is not a clean isolated weight-pack timing.
The optimized follow-up fixes that reporting boundary in a separate attempt.

First FP4 receipt: `results/b200_mlp_gemm/fp4pilot01/fp4/probe.json`, SHA-256
`8d06750ed2f5577b5c608800eec035a067d9a273cf3b0e6c47c815aa3faf4c9b`.
This negative complete-path result motivates a separately recorded fused global
reduction and symmetric CUDA graph replay probe, rather than model integration
or default promotion. Unlaunched BF16 batched-adapter/backward designs were
deferred in favor of the user's FP4 priority; their draft source is retained
under ignored `results/b200_mlp_gemm/deferred_designs/`, not shipped as validated
library implementations.

### Fused tensor-wide scaling and matched graph replay

The second attempt `fp4pilot02/fp4optimized` also completes all twelve cases.
Two Triton reductions replace the Torch amax/inverse sequence; codes, block
scales and inverse scales agree exactly with ordinary reduction for every
sampled shape/path. Ordinary and changed-input graph replay agree exactly
with uncaptured FP4 execution. Both graph legs include the input copy and all
FP4 dynamic scaling, packing, scale initialization, GEMM and output descaling.

| Tokens | Isolated path | BF16 graph, ms | FP4 graph including conversion, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 4096 | Gate/up forward | 0.24565 | 0.17130 | 30.27% |
| 4096 | Gate/up input gradient | 0.30944 | 0.29112 | 5.92% |
| 4096 | Down forward | 0.15358 | 0.16159 | -5.22% |
| 4096 | Down input gradient | 0.13217 | 0.11244 | 14.93% |
| 16384 | Gate/up forward | 0.97028 | 0.56172 | 42.11% |
| 16384 | Gate/up input gradient | 1.17402 | 1.00978 | 13.99% |
| 16384 | Down forward | 0.59644 | 0.53628 | 10.09% |
| 16384 | Down input gradient | 0.50349 | 0.34414 | 31.65% |

Without replay the 16384 gate/up input-gradient path is only 3.66% faster;
the other three full-conversion paths remain slower, and every 4096 path is
slower. This supports an effect of invocation policy and conversion overhead
for this implementation; it does not isolate GPU launch time from memory traffic
or prove end-to-end training gains. No matching model graph integration exists
yet. Tensor-wide dynamic activation scaling can couple rows, so a separate
per-row packing intervention is required before treating this as an isolated
packed-training candidate. This is a precision/data-dependence constraint,
not a relaxation of the existing packing gates.

Optimized receipt SHA-256:
`e8d43e986e1e21f59cbb4b285eb2fd5624ddbe7e86e06da12e2cf8c98048283e`.
The weight-pack timer now ends at the actual packing synchronization: initial
merged-weight preparation is 1.9023 seconds including warm JIT/cache setup,
and down-weight preparation is 0.1575 seconds. These are one attempt's startup
times, not cold-cache or replicated estimates.

### Per-row scaling preserves packed-example independence

The third attempt `fp4pilot03/fp4row` fuses row amax, 16-element E4M3 scaling and
E2M1 packing in one Triton kernel. A BF16 output kernel applies each row's global
scale. Frozen weight scaling remains global and transpose-consistent. This is
an adaptation around NVIDIA's existing GEMM, not a replacement for its MMA core.

Perturbing the first input row by 31.7x changes other rows' outputs by relative
L2 0.08595–0.09068 in the tensor-wide version, on the four 193-token paths.
The per-row version has exactly zero cross-row output effect in every one of
the twelve cases. All native arithmetic and ordinary/changed-input graph checks
pass; maximum decoded-operand relative L2 is 0.00286636. Tensor-wide scaling
therefore cannot be adopted under the current example-isolation contract even
where its graph timings are positive.

| Tokens | Per-row isolated path | BF16 graph, ms | FP4 graph including conversion, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 4096 | Gate/up forward | 0.24620 | 0.18327 | 25.56% |
| 4096 | Gate/up input gradient | 0.31102 | 0.32577 | -4.74% |
| 4096 | Down forward | 0.15619 | 0.19855 | -27.12% |
| 4096 | Down input gradient | 0.13710 | 0.12486 | 8.93% |
| 16384 | Gate/up forward | 0.97375 | 0.61943 | 36.39% |
| 16384 | Gate/up input gradient | 1.17525 | 1.14087 | 2.93% |
| 16384 | Down forward | 0.59883 | 0.64977 | -8.51% |
| 16384 | Down input gradient | 0.50121 | 0.37818 | 24.55% |

This restores isolation but does not establish a uniformly faster FP4 MLP.
In particular, carrying the full padded 9216/18432-element row through one
packing CTA can be costly; this is a hypothesis pending register/spill evidence,
not a demonstrated bottleneck from launch counts. A chunked-row follow-up
separates row reduction from block packing and requires bitwise agreement of
codes, scales and inverse scales against this fused-row implementation.

Third receipt SHA-256:
`6af72596fb1476e9d1f23117cead2a021155f17cf27a95800f70ffa535f693f8`.
The ordinary unscripted per-row complete paths remain slower at both long shapes;
positive entries above depend on graph replay. Dynamic packing/autograd integration,
full LoRA MLP speed, fresh packed-model gates and checkpoint quality remain
unvalidated. No FP4 serving artifact or training-recipe promotion was produced.

### Chunked per-row packing: complete native screen

The fourth attempt `fp4pilot04/fp4chunks` completes all twelve cases. It separates
row amax from 128-block packing chunks while reusing the same pinned NVIDIA
FROST GEMM and output descaling. Quantized codes, E4M3 block scales and row inverse
scales are bitwise identical to fused-row packing at every shape/path. Native
arithmetic, finite outputs, decoded weight-transpose consistency and both graph
replay checks pass. Cross-row perturbation remains exactly zero throughout.
Maximum decoded-operand relative L2 is 0.00286636; original-BF16 GEMM disagreement
is 0.146147–0.146448. These numerical quantities retain their separate meanings.

| Tokens | Chunked-row isolated path | BF16 graph, ms | FP4 graph including conversion, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 4096 | Gate/up forward | 0.24845 | 0.18374 | 26.04% |
| 4096 | Gate/up input gradient | 0.31584 | 0.29287 | 7.27% |
| 4096 | Down forward | 0.16204 | 0.16997 | -4.90% |
| 4096 | Down input gradient | 0.13567 | 0.11967 | 11.79% |
| 16384 | Gate/up forward | 0.97325 | 0.60240 | 38.10% |
| 16384 | Gate/up input gradient | 1.17750 | 1.00523 | 14.63% |
| 16384 | Down forward | 0.59986 | 0.53598 | 10.65% |
| 16384 | Down input gradient | 0.50147 | 0.36154 | 27.90% |

Three paths meet the 5% isolated-path threshold at both long shapes; down forward
still loses at 4096. All ordinary unscripted full-conversion paths are slower
than their paired BF16 legs at both long shapes. The positive results above
therefore require graph replay; the current full-model recipe has not acquired
that integration. Do not add these independent timings into a claimed complete
MLP or optimizer-update speedup, or promote the FP4 recipe based on them.

Triton reports 48/155/246 registers for fused-row packing at widths
2560/9216/18432 and **zero spills**. Chunked packing uses 38 registers, with
16/28/32 for its separate row reduction and zero spills. The large register
footprint is observed; an occupancy bottleneck has not been measured. The
cross-attempt timing trend supports testing this layout but does not isolate
its causal effect; only within-attempt BF16/FP4 legs are paired.

The final library is an explicit CUDA primitive, without registered autograd.
It rejects training inputs requiring gradients under ordinary grad mode so a
caller cannot silently drop them; future training needs an autograd/compiler
wrapper and preserved FP32 adapter masters. Next evidence must cover the
complete LoRA MLP, then fresh full-model packing/isolation/finite-gradient gates,
matched FA4 update timing and quality validation. The BF16 FA4 default remains
unchanged. No repeated FA4 control, full-model FP4 run or teacher call occurred.

Final receipt SHA-256:
`c36750acedb3964228a376732d9cce1255a636769dfcd43e8cbdaadeca9a5ce6`.
All four FP4 attempts, their logs, raw samples and checksum-bound source archives
are collected locally. Every archived source hash matches its launch receipt.
The focused CPU suite passes 27 tests; Ruff and Git whitespace checks pass.
The B200 is idle and remains running in US-NC-2 with the retained workspace and
shared caches. This session has no in-chat scheduling tool, and no after-turn
monitoring is promised.

### Registered native FP4 whole-MLP integration

The complete pilot `integrate01/integrated` installs registered PyTorch custom
operators for frozen native FP4 forward and input-gradient GEMMs, with fake
implementations and explicit autograd. Gate/up forwards share one native GEMM;
all six original FP32 adapter masters retain their identities, state-dict names
and ordinary compiled matmuls. Input gradients include both the native frozen
base and the adapters. Weight pairs are prepared lazily after CUDA placement;
no extra merged BF16 weight buffer is retained. Frozen BF16 originals remain
resident, plus 79,626,256 packed bytes per MLP (about 2.37 GiB across 32 layers).

The three seeded synthetic shapes complete. All output/input/six-adapter
gradients are finite. Native-versus-decoded-FP4 output relative L2 is at most
0.00126408 and each of seven gradient errors is at most 0.00132168, within the
predeclared 1% output / 2% multilayer gradient implementation limits. Perturbing
rows 17 onward by 31.7x has exactly zero effect on rows 0–16. Changed-input and
changed-live-master graph replay agrees exactly with uncaptured compiled native
execution at all shapes. These checks validate the quantized arithmetic and
wiring, not equivalence to the unquantized model. Original-BF16 output error is
24.76–24.82%; gradient errors are 19.91–25.08% on this synthetic fixture.

Matched compiled PEFT and native FP4 complete forward/input-gradient/adapter-
gradient measurements include activation and gradient packing, output descaling,
LoRA casts/matmuls, SiLU and backwards. Both graph legs also include caller input
copies. Six warmups and ten alternating synchronized wall-time samples per leg
are used; no optimizer step or attention is included.

| Tokens | Ordinary BF16, ms | Ordinary FP4, ms | BF16 graph, ms | FP4 graph, ms | Graph time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 193 | 4.84529 | 10.15411 | 0.68211 | 0.65884 | 3.41% |
| 4096 | 4.86652 | 10.08969 | 1.20888 | 1.18998 | 1.56% |
| 16384 | 5.80790 | 10.67218 | 4.68553 | 4.57893 | 2.28% |

The actual complete-MLP gains fail the predeclared >=5% graph improvement rule
at both long shapes. The selector records `not_selected`; no twenty-update
full-model FP4 run, new packing receipt, adapter checkpoint or quality result
is produced. Do not sum the earlier isolated GEMM savings into an update-speed
claim. The conditional full-model entry exists, but remains unvalidated on the
GPU and refuses this pilot. It would request fresh strict 5% packing/gradient,
loss/isolation/finite/memory gates and explicitly record a changed decoder
compilation mode, preserving the separate FA4-specific historical 10% record.

The integrated receipt SHA-256 is
`97d585d8c66b5923ad0a50955a21e802b37a9bd87d9597620eaf51bd7292e0b9`.
Its archived source hashes match the launch receipt. The unchanged B200,
Torch 2.11.0+cu130 and pinned Frontend/FROST overlay reuse persistent compiler
caches. A separate bounded whole-MLP CUDA profile follows the speed failure to
attribute conversion versus native/adapter GEMM cost; profiler event sums are
distinct from the wall-time measurements above.

### Complete-MLP conversion profile

The separate `integrateprofile01/integratedprofile` diagnostic completes both
matched 16,384-token graph traces, ten replay iterations per leg. This fresh
process first compiles at 16,384 tokens; the timing pilot first compiled at 193.
Kernel-duration sums are instrumented observations from this diagnostic, not
an exact decomposition of the pilot's synchronized wall time or a new selection
measurement. The candidate trace contains 58 GPU events per replay, including
four native FROST GEMMs, eighteen ordinary adapter GEMMs, sixteen conversion
kernels and the symmetric caller copy. No additional large BF16 base GEMM is
present in the candidate.

| Candidate GPU work | ms/replay | Share of summed kernel duration |
| --- | ---: | ---: |
| Row reduction, FP4 packing, scale clearing, BF16 output descaling | 1.39850 | 34.12% |
| Four frozen-base native FP4 GEMMs | 0.70543 | 17.21% |
| Eighteen adapter GEMMs, including backward | 0.68510 | 16.71% |
| Compiled elementwise, casts and backward concatenation | 1.28494 | 31.35% |
| Caller input copy | 0.02500 | 0.61% |

Conversion alone costs almost twice as much as the native dense GEMMs in this
trace. Its components are row amax 0.19500 ms, packing 0.80226 ms, output
descaling 0.38900 ms and clearing scale buffers 0.01225 ms. This is measured
evidence that the separate conversion passes are a substantial remaining cost;
it does not prove a particular memory-bandwidth/occupancy limit. The candidate
also has an additional fused backward/concatenation kernel (0.31720 ms), whereas
the compiled BF16 graph lacks that kernel. Native custom-op boundaries and
merged gradient layout are plausible fusion targets, without a measured causal
attribution of the entire integration regression.

Summed kernel durations are 4.61439 ms for the profiled BF16 graph and 4.09897 ms
for FP4. These sums omit inter-kernel and host gaps and use instrumented,
back-to-back replays with a different first compilation shape. Do not substitute
their ratio for the pilot's 2.28% synchronized wall-time reduction or use it to
bypass the failed 5% selection rule. Next useful work would fuse producers with
packing and native GEMM epilogues with output scaling/consumers, while retaining
per-row isolation and live adapters. Simply accelerating the already small FP4
contraction core is unlikely to recover all surrounding cost.

Profile receipt SHA-256:
`19ecb8390868a7d31fd4ba7ddf41291b6508538a97fa48abe19541e7bb7f651a`.
Both raw traces, histograms, derived attribution and checksum-bound source
archives are collected locally; every archive hash matches its launch receipt.
The focused suite passes 37 tests, Ruff and Git whitespace checks pass. The
BF16 FA4 default remains unchanged. The B200 is idle and remains running in
US-NC-2 with retained caches; no after-turn monitoring is promised.

### Hardware conversion preserves arithmetic and improves complete MLP time

The opt-in follow-up replaces the software E2M1 threshold encoder with
Blackwell's `cvt.rn.satfinite.e2m1x2.f32`, using the same low/even and high/odd
nibble convention as the pinned NVIDIA quantizer. It also writes padded scale
slots inside packing instead of separately clearing the scale buffer. Keep the
same row amax, E4M3 block scales, global weight scale, frozen weight orientations,
GEMM tile, FP32 masters and explicit input-gradient quantization contract.
These are implementation changes; neither scale scope nor precision changes.

First attempt `conversion01` passes all nine packing cases, then fails because
the diagnostic decoded oracle still accepts four arguments and the hardware
wrapper supplies five. Preserve the failed receipt as a harness failure. The
corrected `conversion02/conversions` completes all packing and MLP cases.
Codes, the entire scale blob including padding, and row inverse scales match
bit for bit at 193/4096/16384 rows and widths 2560/9216/18432, including zero,
signed-zero and midpoint-tie inputs. Changed-input replay is bitwise identical.
Both packers use 38 registers and zero spills; reduction uses 16/28/32 registers
with zero spills. A register-count change does not explain this improvement.

| Rows | Width | Software pack graph, ms | Hardware pack graph, ms | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 2560 | 0.06118 | 0.05653 | 7.60% |
| 4096 | 9216 | 0.12188 | 0.08632 | 29.18% |
| 4096 | 18432 | 0.21871 | 0.15371 | 29.72% |
| 16384 | 2560 | 0.13817 | 0.10031 | 27.40% |
| 16384 | 9216 | 0.40403 | 0.27343 | 32.32% |
| 16384 | 18432 | 0.76133 | 0.50166 | 34.11% |

These complete pack timings include the same input copy, row reduction, packing
and scale layout; six warmups and ten alternating synchronized samples are
used. The 193-row pack gains range from -0.73% to +0.74%, with tiny absolute
differences. Hardware packing fills padded rows explicitly, so short shapes
need not benefit even though large widths do.

The same process compares complete compiled PEFT BF16, the existing compiled
software-packed FP4 MLP and the hardware-packed MLP. Both FP4 paths share frozen
weights/plans, geometry and original masters. Native output and all seven eager
gradients remain exactly unchanged from the software-packed FP4 path at every
shape. Independent decoded-operand checks, finite/nonmissing gradients, exact
row isolation and changed-input/live-master graph agreement pass. Original-BF16
quantization errors are unchanged; no quality equivalence follows.

| Tokens | BF16 graph, ms | Software FP4 graph, ms | Hardware FP4 graph, ms | Hardware vs BF16 reduction | Hardware vs software reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 193 | 0.68363 | 0.64788 | 0.67148 | 1.78% | -3.64% |
| 4096 | 1.20118 | 1.17759 | 1.05870 | 11.86% | 10.10% |
| 16384 | 4.66143 | 4.55637 | 3.97331 | 14.76% | 12.80% |

Each graph includes caller copies, all dynamic conversions, adapter casts/GEMMs,
SiLU, native input gradients and all six adapter gradients, with six warmups
and ten alternating synchronized samples across all three legs. Ordinary
dispatch still loses to BF16: at 16384 it takes 11.09466 ms versus 5.99310 ms;
software FP4 takes 11.25618 ms. Hardware packing therefore passes the >=5%
complete-MLP graph selection rule at both long shapes, but the ungraphed path
does not provide a training throughput improvement. No repeated FA4 full-model
control is needed; any model screen must reuse its exact frozen contract,
initial adapter and rows, and request fresh precision/packing/memory gates.

Completed pilot SHA-256:
`a32699073a0fc4993d3f1b5328cbcea2f0ca0fcc05bfa0c9b45a372051a8940c`.
Its executed-source archive matches every launch hash. This receipt selects
only the hardware packer, not a fused descaling epilogue or the standard recipe.

### Native row-descaling epilogue preserves raw BF16 rounding

The first epilogue attempt in `conversion02/epilogue` fails before timing because
the already planned projection graph is immutable. The corrected
`descale01/epilogue` constructs a fresh graph with exactly the original operand
dimensions, strides, FP4/E4M3 formats and selected FROST tile configuration.
It declares the raw matmul result BF16 while virtual, then multiplies by a
per-row FP32 scale and emits BF16. A small row-scale kernel computes only
`1 / (inverse_a[row] * inverse_b)`; the full matrix output-descaling pass and
its additional matrix buffer are removed. This uses NVIDIA's existing pointwise
epilogue compiler and MMA core, without a kernel-template fork or dtype change.

All twelve cases complete with bitwise-identical outputs to the original
separately-descaled GEMM. Finite/nonzero outputs, exact row isolation and changed-
input graph replay pass throughout. Both legs use hardware packing and the same
original weight pair and native tile. Timings include conversion, row scales
and symmetric caller copies, six warmups and ten alternating synchronized
samples. These are isolated projection paths, not a complete MLP or model update.

| Tokens | Path | Separate descaling, ms | Fused epilogue, ms | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 4096 | Gate/up forward | 0.17350 | 0.12305 | 29.08% |
| 4096 | Gate/up input gradient | 0.23136 | 0.22643 | 2.13% |
| 4096 | Down forward | 0.13333 | 0.12820 | 3.85% |
| 4096 | Down input gradient | 0.11499 | 0.09076 | 21.08% |
| 16384 | Gate/up forward | 0.56384 | 0.37151 | 34.11% |
| 16384 | Gate/up input gradient | 0.75335 | 0.72440 | 3.84% |
| 16384 | Down forward | 0.41067 | 0.38396 | 6.50% |
| 16384 | Down input gradient | 0.33121 | 0.23711 | 28.41% |

The wide outputs benefit most from eliminating a matrix pass; this timing
pattern supports the fusion but does not itself measure memory-bandwidth
saturation. At 193 rows, one path improves 4.46% and the other three regress
3.71–9.09%; keep the short-shape results. Do not add independent path timings
into a claimed optimizer-update improvement. The combined MLP follow-up must
retain complete input/adapter gradients and compare against packing alone,
software FP4 and BF16 before selecting the full-model intervention.

Receipt SHA-256:
`2efae72f96d15858b07089e68196b82ffd15f8568db36e6df6b804d3bdb048d5`.
All archived source hashes match the launch receipt. Preserve the earlier graph-
mutation failure separately; it is an API-construction failure, not arithmetic
evidence against the corrected epilogue.

### Combined conversion optimizations: complete MLP selection

`fusedmlp01/fusedmlp` registers the row-descaling native GEMM in both frozen
forward and input-gradient operators. Retain the software and hardware-only
operators as independent controls; packed weights are shared and plans are
keyed separately by epilogue mode, device and geometry, with the live caller
stream supplied on every launch. Keep original parameter identities and all
six FP32 adapter masters. The default profile is untouched.

All nine packing checks and all three complete-MLP cases pass. The combined
output and every one of seven gradients are exactly identical to the previous
software-packed FP4 MLP. Native-versus-decoded implementation errors stay below
0.14%; finite/missing-gradient, exact row isolation and changed-input/live-master
graph checks pass. Original-BF16 quantization disagreement remains unchanged.
These are arithmetic and wiring checks, not model-quality acceptance.

| Tokens | BF16 graph, ms | Software FP4 graph, ms | Hardware-only FP4 graph, ms | Combined FP4 graph, ms | Combined vs BF16 reduction | Combined vs hardware-only reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 193 | 0.68228 | 0.66572 | 0.67311 | 0.66763 | 2.15% | 0.81% |
| 4096 | 1.20102 | 1.17731 | 1.06061 | 0.97835 | 18.54% | 7.76% |
| 16384 | 4.61722 | 4.35546 | 3.82395 | 3.48949 | 24.42% | 8.75% |

All four legs are measured in the same process with complete MLP forward,
input/adapter backward, dynamic packing/scaling and caller copies. Six warmups
and ten alternating synchronized wall-time samples are used. Within-attempt
legs are matched; do not treat timing differences between separate attempts as
an isolated causal effect. Ordinary dispatch remains slower than BF16: at
16384 the combined path takes 10.97324 ms versus 5.97418 ms; hardware-only takes
10.97040 ms. Whole-model graph integration is therefore essential to realizing
the local gain, and these data cannot establish an optimizer-update improvement.

The combined candidate passes the predeclared >=5% improvement rule at both
long shapes and beats hardware packing alone. Select it for a bounded fresh
strict model screen; the selector requires matched four-way timings and exact
operand/MLP arithmetic evidence. The twenty-update screen uses the immutable
320-row FA4 cohort, targets and initial master, while recording both conversion
flags and the changed `reduce-overhead` decoder compile mode. Request all fresh
precision/packing/finite/memory gates; no startup-receipt reuse or widening of
the 5% limit occurs. A failed gate stops before optimizer updates.

Completed combined pilot SHA-256:
`1710694a780a1062b7d034903e422eac148c98ae61f6f28602bcda0c8f6e1aa7`.
Its archive matches every launch source hash. Fifty focused CPU tests pass,
including three registered autograd/compiler paths, optimizer updates on live
FP32 masters and fail-closed candidate selection; Ruff and whitespace checks pass.

### Full-model conversion screen fails fresh packing parity

`trainconv01` launches the selected combined pilot, installs both optimizations
in all 32 MLPs and verifies the checksum-bound initial FP32 master. It stops at
the fresh eager packing gate before decoder compilation, longest-batch memory
preflight or optimizer updates. Preserve this negative receipt; no complete
update timing, trained adapter or quality result exists for this attempt.

All three isolation cases pass exactly: repeat/perturb decision drift and cross-
example input gradients are zero, with finite nonzero own-example gradients.
The equal-example-weighted independent loss is 1.11006910 and packed loss is
1.06096554. The absolute difference, 0.04910356, exceeds the unchanged permitted
`0.02 + 0.02 * abs(independent_loss)` bound of 0.04220138. Adapter-gradient
relative L2 is 0.67898673 (67.90%), exceeding the predeclared 5% limit. The
FA4-specific historical 10% acceptance does not cover this new FP4 recipe.

Layer diagnostics are exactly equal through decoder layers 0–2. The first
observed difference appears at layer 3, the first full-attention layer (1.94%
relative L2), then reaches 23.08% at layer 31 and 32.17% at the LM head. This
locates the onset but does not isolate the attention versus MLP contribution
inside layer 3. Amplification of packing-dependent rounding by repeated FP4
quantization is a plausible next hypothesis, not an established cause. The
bitwise conversion/epilogue proofs remain valid for the tested operand and MLP
fixtures; they do not establish full-model packing or BF16 quality equivalence.

Separately, the original-FLA versus FlashQLA canary has finite gradients but
fails strict parity at 102.16% gradient relative L2. It is recorded under the
existing explicit `selected_finite` FlashQLA policy; that acceptance does not
waive the packing loss/gradient gate which stops this run. Do not report either
failed strict receipt as a fresh numerical pass.

The final runtime receipt records 64 frozen forward/transposed weight pairs,
30 plans and 2,548,040,192 packed-weight bytes (2.37 GiB). All 28 archived source
hashes match the launch receipt. Results and logs are collected locally; no
model updates or unchanged strict reruns occur. BF16 MLPs with FA4 remain the
standard. The B200 is idle and remains running in US-NC-2 at $6.79/hour;
the network-volume compiler caches are retained. Active-turn monitoring ends
with this completed screen; this session has no recurring agent scheduler.

Final receipt SHA-256 values:

- `trainconv01/summary.json`:
  `0062d15be9342e2c369b6b0495644ea230242c1bb8129a440adebc311b98844e`.
- `trainconv01/causal_adapter/packing_canary.json`:
  `bda454f82233e43ce1da0bfa82849c85a0e39f3d0cd4304431d5c38fd3ba5324`.
- `trainconv01/causal_adapter/gated_delta_canary.json`:
  `3d571442ef89653b3c8a3b92c9083ee9a36161868c6a524cda42b5fddfd17df6`.
- `trainconv01/native_runtime.json`:
  `23c70ab7b9b3298ec78f79886e214a656843d30176f68832c0cc1ef1eda26450`.

The user subsequently explicitly requests end-to-end timing despite the gradient
disagreement. A separate bounded timing-only follow-up retains both failed
strict receipts and records that new authority. Waive only loss/relative-gradient
parity for speed measurement; require fresh finite/isolation, compiled replay
and memory checks, preserve FP32 masters and the unchanged physical contract.
Keep the selected combined FP4 pilot, twenty updates (ten warmup/ten measured),
historical FA4 control and shared caches. Report any resulting gain as speed
evidence for the combined FP4/compile recipe, not learning/quality acceptance.

`traintiming01` records the requested timing-only authority and accepts the
finite, exactly isolated eager packing receipt for timing, while strict and
learning parity remain false (68.11% gradient relative L2). It then fails before
updates during the compiled logit canary: a decoder graph-segment output is
overwritten when CUDA-graph iteration tracking advances between segments.
Preserve this integration failure separately from numerical parity. Its final
summary SHA-256 is
`db23cb1b08bdf316509e618423f867e9fade2a4f9d8f1034a6efcd5ab0c3f92b`;
eager packing receipt SHA-256 is
`fed2055a68ad35da44b558fb642ab817f7d7ca1253c2fcedf0112022960d8db1`.

The retry adds one root-model pre-forward hook calling the supported
`torch.compiler.cudagraph_mark_step_begin()` API, so all decoder segments in
one physical model invocation share a graph iteration. Mark once per physical
forward, never once per decoder layer or before its pending backward. No output
clone or arithmetic change is added. This follows the
[PyTorch 2.11 iteration-boundary guidance](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.cudagraph_mark_step_begin.html).
Fourteen timing-path CPU tests pass, including single root marks through nested
layers, live parameter/input gradients and unchanged parameter/state identities.
Use the same pilot, authority, fresh checks, twenty updates and retained caches;
the GPU retry must establish that the boundary fix actually resolves replay.

`traintiming02` passes the previous graph-lifetime failure and reaches gradient-
enabled packed variants, then encounters a native `kernel_py` segfault in the
autograd worker before completing the compiled packing receipt or any updates.
The native log alone does not identify which kernel caused it. Preserve the
sources, eager receipt and logs; do not treat successful no-grad replay as
successful graph-enabled backward. The process remains in exit handling and is
sent SIGTERM and SIGKILL after artifact collection, but still retains 12.67 GiB
of device memory while exiting. No compiler descendants remain. A fresh CUDA
compute/synchronize probe succeeds with a sum of 16.0; the GPU is usable for a
separate process. At the subsequent `traintiming03` startup check, the old
context has released its memory and device allocation is zero before new model
placement. No residual-context memory confound remains for that measurement.
No timing result exists for either graph-enabled attempt.
The graph retry's final launcher return code is `-9` after forced cleanup;
summary SHA-256 is
`aa1b742b957250766f6893d8aa9ed3b681217febc5abae78554af41c404e9663`.

To fulfill the requested end-to-end timing, select `--compile-mode default`
for a separate bounded follow-up with the same two conversion optimizations,
pilot, authority, cohort, master, physical batching and fresh checks. This
removes graph-enabled decoder execution and matches the historical control's
compile mode. The local whole-MLP graph speedups no longer predict this path;
report any regression openly rather than extending graph-only numbers to full
training. The root physical-forward marker can remain but no decoder graphs
are enabled by the default mode.

### Completed matched standard-mode end-to-end timing

`traintiming03` completes all twenty updates with both conversion optimizations
installed in all 32 MLPs and `selective_torch_compile_mode=default`, matching
the historical FA4 control's mode. The unchanged initial FP32 master, all 147
physical partitions, logical indices, tokens and padded-token fields match the
control exactly. No repeated FA4 control run occurs. Missing/nonfinite-gradient
checks remain enabled through every update; all losses/gradients are finite.

| Quantity | Historical BF16 FA4 control | Combined native FP4, default mode |
| --- | ---: | ---: |
| Measured mean, updates 11–20, seconds | 4.08648 | 19.42944 |
| Measured total, seconds | 40.86484 | 194.29436 |

The native FP4 trajectory takes 4.75456x as long (375.46% more time), so it fails
the five-percent speed improvement rule. Its Trainer loop takes 436.2034
seconds. Preserve all raw step durations; do not discard slow measured updates.
The ten measured durations range from 8.01514 to 34.59060 seconds. This result
is observed end-to-end wall time for the predeclared trajectory, including
first-use planning/compilation; it is not a fully warmed contraction benchmark.

There are 94 distinct physical token shapes across the twenty updates. The
warmup half has 67 and the measured half has 69, of which 27 are new relative
to warmup (28 of 73 measured physical batches). Cache inspection during the run
shows new `_pack_row_blocks` and `_row_scale` IR/PTX/cubin artifacts. These
kernels specialize on row-dependent group/row counts, and the native runtime
ends with 416 geometry-specific GEMM plans. Ten updates therefore do not warm
the candidate's entire measured shape envelope. Timing alone does not apportion
the observed regression among compilation, plan construction, Python dispatch,
conversion and contraction. The prior matched ordinary-dispatch complete-MLP
pilot also loses to BF16, so removing first-use cost alone does not establish a
training speedup. The supported next targets are runtime row-count arguments
in conversion kernels, reusable native plans and reliable MLP graph integration.

The longest-batch preflight passes on the actual 32 longest training inputs,
maximum length 28733, with finite gradients and unchanged master. Peak allocated
memory during the trajectory is 148.30696 GiB. The runtime retains 64 packed
weight pairs totaling 2,548,040,192 bytes; no extra merged BF16 copy is retained.
The final adapter contains 256 FP32 tensors and 679,511,752 bytes. Its master
digest changes to
`6b6d8eec418209c61b9bbbd3a4b6a6498d55ee861061c9bf5cd1b35368a94d56`.

Strict/learning packing acceptance remains false in both receipts: eager
gradient relative L2 is 68.11% and compiled is 65.09%; both loss comparisons
also fail their original bound. Exact isolation and finite gradients pass,
with only loss/relative-gradient checks waived under the quoted user timing
authority. This run establishes timing and operational progress, not BF16
quality equivalence or held-out monitoring performance. Keep BF16 MLPs with
FA4 as the standard; retain all opt-in conversion implementations and failed
graph/strict receipts.

All 29 archived source hashes match the launch receipt. The completed model,
checkpoint, reports and logs are collected locally. The process exits with
return code zero and the GPU allocation returns to zero. The B200 remains
running and idle in US-NC-2 at $6.79/hour, with its network-volume caches intact;
active-turn campaign monitoring is complete.

Final standard-mode receipt SHA-256 values:

- `traintiming03/summary.json`:
  `89f4cec6f6aed5bff02e4e9cfb56b5e2ff5f18049be2687c437853a6559931ee`.
- `traintiming03/causal_adapter/training_metadata.json`:
  `f1b7bfed8fa3301002aba91fce991bc39dfee5c587c9a25fbb2f3257efd09079`.
- `traintiming03/causal_adapter/packing_canary.json`:
  `9ec1b04890820ec9bc0047e2df4b3c5ba8dfa12e0c89841724a9d1383c0bfe14`.
- `traintiming03/native_runtime.json`:
  `3b2c97704cc65d460d14628f1d3d91b8600b716afcb89b24500c8a81f04e2c29`.
- `traintiming03/causal_adapter/adapter_model.safetensors`:
  `cdae0133aeff9a4b7eb03cdc87008d03e5a5b40e7abb7ce1466821b3fdc28445`.

### Exact-shape warmed FP4 training

The user requests warmed performance on 2026-10-05 after the preceding default-
mode run continued first-use specialization during measured updates. `warmed03`
completes the same twenty-update trajectory in `default` mode with unchanged
conversion kernels and retained cache namespaces. Before optimization, two
same-process passes replay the full twenty logical batches with full backward,
no optimizer updates, preserved RNG/sampler state and unchanged FP32 masters.
Both replay physical contracts match all 147 historical FA4 partitions exactly.
The first pass creates 248 native plans and 265 Triton specializations; the
second creates zero plans, specializations, Dynamo graphs or Inductor graph-cache
misses. No third warmup is needed. All twenty actual optimizer updates also show
zero increments in each of these counters. These Triton events count first-use
in-process specialization, including loading already compiled persistent entries;
they are not all fresh binary compilations.

| Complete update, updates 11–20 | Mean seconds | Difference versus historical BF16 FA4 |
| --- | ---: | ---: |
| Historical BF16 FA4 control | 4.08648 | Reference |
| Combined native FP4, first-use shapes included (`traintiming03`) | 19.42944 | 375.46% more time |
| Combined native FP4, all measured shapes warmed (`warmed03`) | 3.74480 | 8.36% less time |

The ten warmed measured updates total 37.44805 seconds versus 40.86484 for
the historical control. Keep all samples; their range is 2.40343–5.39594 seconds.
The warmed candidate clears the predeclared five-percent timing improvement
rule. This is comparison with the immutable historical control, without fresh
same-runtime control replication or uncertainty estimates. It supports warmed
execution follow-up, not automatic recipe promotion or full-corpus wall-time
savings: unseen packed shapes still incur preparation. The earlier ordinary-
dispatch standalone MLP pilot remains a separate, slower result; it is not
substituted for this full-model measurement.

Warmup forward/backward durations sum to 204.84156 and 73.96002 seconds for the
two passes. Receipt writes, batch collection and verification are additional
warmup overhead. The Trainer loop's 363.3807 seconds includes the warmup callback;
use synchronized optimizer-step durations, not that loop total, for warmed
throughput. Native runtime ends at 416 plans and 64 packed weight pairs totaling
2,548,040,192 bytes. Peak allocated memory is 148.30699 GiB. The longest-batch
preflight passes, and every actual update has finite loss/nonmissing gradients.

The initial master remains
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
The final master digest and 679,511,752-byte saved adapter are identical to
`traintiming03`, with all 256 tensors FP32. This confirms that extra warmup
preserves this optimization trajectory. Strict/learning packing acceptance
remains false, with eager/compiled gradient relative L2 68.11%/65.09% and failed
loss bounds. Exact isolation and finite checks pass. Retain the original user
timing-only authority; no BF16 quality equivalence or held-out evaluation occurs.
BF16 MLPs with FA4 remain the selected standard.

The initial `warmed01` launch fails because the wrapper precreates the runner's
exclusive log directory; `warmed02` then fails Hydra parsing of punctuation added
to the authority note. Neither reaches optimizer updates. Preserve both failures;
the corrected wrapper uses a separate launcher directory and the original
validated authority string. Sixty-three focused CPU tests pass, plus Ruff and
whitespace checks. The executed warmup source is archived with all 31 launch
source hashes verified; subsequent interface annotations do not change execution.
The successful process exits zero and the GPU becomes idle. Its checkpoint,
adapter, reports and logs are collected locally. The B200 remains running in
US-NC-2 with its volume/caches; active-turn monitoring is complete.

Warmed receipt SHA-256 values:

- `warmed03/summary.json`:
  `8a6900a482fce0568d40d631f2f1342a845cd9e096ebc8c897bcaad5a3fd0f13`.
- `warmed03/shape_warmup.json`:
  `506af5051d5c152b635394abe9bee2fb46346e73c8ea78baace35faf14dfb60d`.
- `warmed03/causal_adapter/training_metadata.json`:
  `14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb`.
- `warmed03/causal_adapter/packing_canary.json`:
  `9ec1b04890820ec9bc0047e2df4b3c5ba8dfa12e0c89841724a9d1383c0bfe14`.
- `warmed03/native_runtime.json`:
  `3b2c97704cc65d460d14628f1d3d91b8600b716afcb89b24500c8a81f04e2c29`.
- `warmed03/causal_adapter/adapter_model.safetensors`:
  `cdae0133aeff9a4b7eb03cdc87008d03e5a5b40e7abb7ce1466821b3fdc28445`.

### Current warmed full-model profiling

On 2026-10-05, `warmedprofile01` profiles fixed actual optimizer updates 11,
15 and 20 of the combined native FP4/FA4 implementation. The twenty-update,
320-row cohort, initial FP32 adapter master, objective, default compile mode
and all 147 physical partitions match the historical control. No fresh BF16
control is run. Startup checks are reused from checksum-bound `warmed03`, with
hardware/software and native-source identity verified. Original strict failures
remain in the receipt with `performed_this_run=false`; numerical, isolation and
longest-batch probes are not repeated. Finite/missing-gradient update checks
remain active. This is a diagnostic campaign, not a new speed or quality screen.

Two exact-shape forward/backward replay passes take 531.35439 and 72.37660
seconds in summed step time. The first adds 376 native plans, 447 in-process
Triton specializations and eight Dynamo graphs, with zero Inductor graph-cache
misses. Its first batch takes 272.93813 seconds; the reused diagnostics did not
prepare the ordinary training variant. The second pass and all twenty actual
updates add zero plans, specializations or compiler graphs. Warmup preserves
masters, RNG/sampler state and empty optimizer state. The runtime ends with
376 plans and 64 packed weight pairs totaling 2,548,040,192 bytes. Unlike the
earlier run, this process does not build the extra diagnostic-shape plans.
Persistent compiler/kernel cache paths and pinned runtime versions are unchanged.

The table pools **summed CUDA kernel time** across the three selected updates.
GPU annotation events and CPU operators are excluded to avoid double counting.

| Identifiable GPU work | Share of summed kernel time |
| --- | ---: |
| FlashQLA/GDN scan, triangular solve, convolution and normalization | 24.93% |
| Ordinary GEMMs, including LoRA and other projections | 20.17% |
| BF16 causal variable-length FA4 | 14.24% |
| SiLU and fused pointwise work, without full module attribution | 10.02% |
| Tensor copies and dtype conversions | 8.12% |
| Frozen-base FP4 GEMMs with fused output descale | 5.65% |
| Dynamic FP4 input/gradient conversion | 4.31% |
| Other/unclassified kernels | 12.56% |

The GDN subtotal is 21.55% scan/solve/convolution and 3.38% normalization.
The traced `tilelang_kkt_solve_kernel` is verified against the pinned FlashQLA
GDN source. Generic SiLU fusions are not all assigned to MLPs: GDN shells also
contain SiLU work. Ordinary GEMMs are not all adapters or MLPs. Broad matrix
and pointwise attribution requires additional module/correlation evidence.

| Update | CUDA kernels | Summed kernels, s | Device interval union, s | First-to-last device span, s | No device event inside span, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 37,742 | 3.72560 | 3.72114 | 4.63969 | 0.91854 |
| 15 | 46,144 | 4.95855 | 4.95234 | 5.75908 | 0.80675 |
| 20 | 33,520 | 4.55159 | 4.54740 | 4.96685 | 0.41945 |

Device union includes memcpy/memset and merges overlapping intervals. Gaps
average 0.71491 seconds, 13.96% of the pooled device span. There are 196–268
`cudaStreamSynchronize` calls per traced update, with 0.571–0.929 seconds of
summed CPU duration. CPU launch, transfer and synchronization durations overlap
GPU execution and must not be added to these GPU totals. Missing device events
suggest dispatch/synchronization investigation but do not alone prove a CPU
bottleneck. Instrumentation itself can increase gaps and launch costs.

The current full-model conversion share is much smaller than the earlier
34.12% unfused synthetic-MLP share; their scopes and kernels differ, so that is
not a matched before/after reduction. Frozen FP4 contractions plus their dynamic
conversion now occupy only 9.96% of identifiable total kernel time. Larger
supported investigation targets are GDN backward/convolution, remaining BF16
projections, and launch/copy overhead. Further core FP4 GEMM tuning alone has
limited room in this profile; no new implementation or speedup is established.

Instrumented update wall times are 4.84032/5.82623/5.04231 seconds. Keep the
unprofiled 3.74480-second result as the speed receipt; selected trace shares
cannot be multiplied by that mean to claim unprofiled component wall times.
The profile run's loss history and final adapter are **not identical** to
`warmed03`, despite matching initial masters and physical batches. Startup and
compilation history differ; the cause of numerical trajectory drift is not
isolated here. Treat these as diagnostic traces of the same implementation,
not exact replay of the earlier optimization trajectory. No new loss-parity,
held-out-quality or default-promotion claim follows.

The process exits zero after twenty finite updates; peak allocated memory is
148.25233 GiB. The final saved adapter has 256 FP32 tensors and 679,511,752
bytes. All 33 archived launch-source hashes, three raw-trace hashes and the
separately recorded post-launch analyzer hash are verified. Raw traces, reports,
checkpoint and logs are collected locally. The analyzer's nine focused tests,
Ruff and whitespace checks pass (the initial launch feature also passed 54
unique relevant CPU checks). The B200 remains running and idle in US-NC-2;
active-turn monitoring is complete. BF16 MLPs with FA4 remain the standard.

Current profiling receipt SHA-256 values:

- `warmedprofile01/summary.json`:
  `64507ef439da31bb4294b54cc3f6a310b3e4e662d2a4fe224f3f493cba07c772`.
- `warmedprofile01/causal_adapter/training_metadata.json`:
  `b3215136208889c622b4449751f08ece1a54a7ddb995d42bd10719710cb928e3`.
- `warmedprofile01/warmed_profile/analysis.json`:
  `9c01df2de406f3debd117c8686237fdbf543113e5a5459e27660c678eb9993b7`.
- `warmedprofile01/causal_adapter/adapter_model.safetensors`:
  `0b9ea36eb19d012c730107ae222b490b4ad10620c795646d93cae8f94e68b71e`.
