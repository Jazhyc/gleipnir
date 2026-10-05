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

### Resident FP4 baseline and projection attribution

On 2026-10-05 the user selects native FP4 MLPs plus BF16 FA4 as the timing
baseline and requests a persistent training worker, including future sessions.
Session `resident01` on the existing US-NC-2 B200 keeps PID `10916`, the loaded
model, compiled modules, 376 native GEMM plans and 64 packed weight pairs
(2,548,040,192 bytes) alive after its initial queue. It reuses network-volume
compiler caches and the checksum-bound `warmed03` startup receipt. Original
strict numerical failures remain explicit; no numerical, isolation or largest-
batch probes are repeated. Actual updates retain finite/missing-gradient checks.

The worker prepares the twenty benchmark shapes once without optimizer updates.
Summed preparation time is 521.54616 seconds, including a 267.909-second first
batch. This is process preparation despite persistent disk caches; subsequent
compatible trials retain these in-process resources. All sixty actual updates
across the initial queue add zero plans, Triton specializations, Dynamo graphs
or Inductor graph-cache misses. No second full preparation replay is needed.

| Resident trajectory | Mean synchronized seconds/update, updates 11–20 | Trial wall seconds |
| --- | ---: | ---: |
| `01baseline`, uninstrumented | 3.65854 | 605.36209, including preparation |
| `02repeat`, uninstrumented | 3.67101 | 83.44038 |
| `03gemmprofile`, instrumented at update 15 | 3.78291 | 138.85837 |

The two uninstrumented means differ by 0.341%; their pooled mean is 3.66478
seconds over twenty measured updates. Keep all samples; their individual ranges
are 2.36495–5.27907 and 2.35135–5.29009 seconds. Use this resident baseline for
future matched optimization trials, rather than repeating the historical BF16
control. The profiled trajectory is diagnostic, not a third speed measurement.
Trial wall time excludes initial imports/model loading; it includes state reset,
the training call, final-master hashing and adapter export. Profiler
export is also included.

Each trial restores the initial FP32 adapters, resets AdamW/scheduler state and
RNG, and reproduces the historical 147 physical partitions. All three trajectories
have identical loss/gradient-norm/learning-rate logs and final-master hashes.
`reset_validation.json` verifies the two baseline resets exactly. Initial master
is `a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`;
final master is `cf38e3e6881cebba402f3d16a1ed2d31dd3a259876375e887b2821cc9426fe3f`.
All three saved adapters are byte-identical to `warmedprofile01`, with 256 FP32
tensors and 679,511,752 bytes. This establishes reset correctness within the
resident recipe, not equivalence to BF16 or the distinct `warmed03` trajectory.

Update 15 records operand shapes, CPU/GPU External IDs and autograd sequence
numbers. Forward regions are identified from actual FlashQLA/FA4 calls; sequence
numbers link both compiled and ordinary `MmBackward0` nodes to their forward
regions. This resolves shared GDN/full-attention weight shapes without assigning
all ordinary GEMMs to adapters. GPU annotation and CPU durations are excluded.

| Identified work in update 15 | Share of summed CUDA kernel time |
| --- | ---: |
| GDN scan, solve, convolution and normalization | 25.24% |
| Frozen GDN projection GEMMs | 9.67% |
| LoRA GEMMs, including adapter weight gradients | 6.62% |
| Frozen full-attention projection GEMMs | 2.76% |
| Language-model head GEMMs | 1.31% |
| Unattributed ordinary GEMMs | 0.0013% |
| BF16 FA4 | 13.62% |
| Frozen FP4 MLP GEMMs and dynamic conversion | 9.99% |
| SiLU/fused pointwise work without full module attribution | 10.06% |
| Copies and casts | 8.14% |
| Other/unclassified kernels | 12.60% |

Frozen GDN projections are 47.50% of ordinary GEMM time, compared with LoRA's
32.50%. The single fused GDN backward kernel accounts for 569.196 ms, 11.47% of
total kernel time; it is the largest individual kernel. GDN preparation and
causal convolution are additional work. The trace launches 46,144 CUDA kernels,
with 4.96062 seconds of summed kernel duration and a 5.96702-second device span.
The 1.01226-second gap without device events is instrumented evidence only.
Do not multiply these shares by the unprofiled mean to claim component wall time.

The next practical GEMM intervention is the large frozen GDN `in_proj_qkv`,
`in_proj_z` and `out_proj` contractions, reusing the existing NVIDIA NVFP4
packing/GEMM/descale machinery for forward and input gradients. Leave small
`in_proj_a`/`in_proj_b` gate projections in BF16, recurrent Q/K/V in BF16 and
gates/normalization in FP32. Investigate sharing input packing between QKV and Z
projections, rather than duplicating conversion. This is a proposed intervention,
not an implemented speedup or new numerical acceptance. Check its changed
arithmetic once, then measure complete updates against the resident FP4 baseline.
The largest broader target remains GDN backward/scan/convolution; core MLP FP4
GEMM tuning has less room. No held-out quality promotion follows.

All 39 initial launch-source archives are hash-verified. The loaded worker also
archives its exact source and records SHA-256
`c47c8292e9aa26443c2069ea30ff68102020369dafe6c943793892f92b611704`.
This differs from its initial launch archive because candidate hot-loading support
was added before the worker module imported. The stdlib-only control refactor and
post-launch analyzer are separately recorded; the analyzer records its own hash.
Fifty-five unique focused CPU checks pass across the completed feature and
attribution work, with Ruff and whitespace checks. Adapters, trace, receipts and
logs are collected locally. The queue is complete and the worker remains alive,
idle, with the model/caches resident. There is no in-chat scheduling tool; no
after-turn heartbeat monitoring is promised. See the
[resident-worker decision](../decisions/b200_fp4_optimization_worker.md).

Resident receipt SHA-256 values:

- `01baseline/receipt.json`:
  `ef3ef71321ea77f2cd486c50b1df28b78f89a9b7743ed4cc1fe97df132af378a`.
- `02repeat/receipt.json`:
  `739ad4e505d42de98365db81d748faee9996badb03a09876146fecc1e27f143b`.
- `03gemmprofile/receipt.json`:
  `d3e4a59a56ff454c62b49d8e1735ce0d937a25ce7b2571cfbb166b949e00488f`.
- `reset_validation.json`:
  `20e778ceefa018f3c751392b82a7389a7f6c30f5267ce2a2bae19960324675c3`.
- `03gemmprofile/gemm_trace.json`:
  `2146c169380735a28237a061e1bddaa079aa158f7b108758d9d523b247f884bd`.
- `03gemmprofile/gemm_analysis.json`:
  `916fe2811508cd58e2a5e351c09e7770f9a7105af1c3d8467762cb1e68c34f56`.
- All three `adapter/adapter_model.safetensors` files:
  `0b9ea36eb19d012c730107ae222b490b4ad10620c795646d93cae8f94e68b71e`.

## Resident frozen-GDN NVFP4 projection screen, 2026-10-05

Both merged QKV/Z packing and separate QKV/Z projections lose to the FP4-MLP,
BF16-GDN baseline. The merged path saves input packing and combines the input
gradient contraction; the separate path removes that integration choice. Both
use the existing NVIDIA hardware NVFP4 packing/GEMM/fused-descale operations,
including the frozen output projection. Small gate projections, BF16 recurrence,
FA4, FP32 gates/normalization and adapter masters remain unchanged.

After the original installer fails before preparation or updates, the corrected
worker `resident02` on the same B200 establishes two matched controls before
either candidate. Their pooled mean, 3.67836 seconds, replaces the prior worker's
reference for the unchanged, predeclared 2% selection rule.

| Trial | Warm seconds/update, updates 11–20 | Extra time versus pooled control | Mean training loss, updates 1–20 |
| --- | ---: | ---: | ---: |
| FP4 MLP / BF16 GDN control | 3.68226 | +0.11% | 0.50287 |
| Exact control repeat | 3.67445 | −0.11% | 0.50287 |
| FP4 GDN, merged QKV/Z | 4.16069 | +13.11% | 0.49842 |
| FP4 GDN, separate QKV/Z | 4.04658 | +10.01% | 0.45917 |

All ten measured updates of each candidate are slower than their matched pooled
controls. All eighty actual updates have finite/nonmissing gradients and add
zero native plans, Triton specializations, Dynamo graphs or Inductor graph misses.
All four runs reproduce the same 147 physical partitions and initial master
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
Both controls exactly reproduce the prior worker's loss/gradient-norm/LR logs
and final master. Thus compilation during measurement does not explain the loss
of speed, and merging is not its sole cause. No new candidate trace isolates
conversion, eager dispatch or individual contraction costs.

On the first matched logical batch, control loss is 0.52186; merged/split losses
are 0.55682/0.61754, absolute differences 0.03496/0.09568. Relative adapter-gradient
L2 differences are 117.84%/137.15%. These are finite timing-only screens, with
numerical equivalence explicitly unclaimed. Lower twenty-update mean losses do
not establish held-out quality or better convergence. Reject both candidates for
the timing baseline; retain FP4 MLPs with BF16 GDN and BF16 FA4. The general
quality-validated BF16 profile and historical strict failures remain separate.

The replacement worker primes once (524.79032 seconds of step time); candidate
preparation takes 367.45113/245.56307 seconds wall time with no optimizer updates.
The control repeat takes 83.64615 seconds including reset/export. Exact-source,
variant, worker-PID, master, physical-contract and completed-warm-receipt checks
now permit reusing candidate validation within this worker. Eight CPU identity
tests pass; no repeated GPU candidate was launched to exercise receipt reuse.

Four FP32 adapters, receipts, source archives and logs are collected locally
under `results/b200_mlp_gemm/resident02/` and the matching logs directory.
Each adapter has 256 FP32 tensors and 679,511,752 bytes. Forty-two launch-source
archives and separately executed candidate/library sources are checksum checked.
The executed worker SHA-256 is
`e3e8fbb0ec4c35fba5c72501ec17fc67168dffae84a9c7fea11e6801b1670042`;
both candidate library archives have SHA-256
`fa7d65d2803afa40ad065354ed6ed279fca5076f37ac63d05126473abb1f6056`.
Twenty-eight focused CPU tests cover the integration, recovery and receipt reuse.
The completed queue leaves PID 11905 alive and idle with the baseline GDN methods
restored, model/caches resident, and no promised after-turn heartbeat.

Receipt SHA-256 values, in table order:

- `04e2274db5ac8fc17fb8a3b986e4097fa54b1937082503658fbf8623c8dcc676`.
- `21a4d7514b7af1cb13f8a3420eb5181b23dd04f62ed28220cea364cdb9485670`.
- `50be0a66cfbe0437c822b6bb7c8c6b5eff7835e2ff68db101ba92851a68e251a`.
- `2d976b3f4790dba91abb428c7e87bee943b0d9e4cf24c6a26f736a60ee37ce8b`.

### NVIDIA BF16 causal Conv1D follow-up

The user redirects investigation toward a smaller BF16 convolution replacement.
NVIDIA's [frontend roadmap](https://github.com/NVIDIA/cudnn-frontend/issues/442)
describes native width-four SiLU forward/backward and packed sequences. Inspection
of our already-pinned frontend source confirms `cudnn.ops.causal_conv1d` routes
channel-last BF16 inputs and contiguous width-four filters to the native training
backend. It accepts CUDA int32 `cu_seqlens` for packing but explicitly rejects
`seq_idx`; reuse our existing cumulative packing offsets rather than infer them
on the host or omit sequence isolation. The GDN disabled-forward boundary is
compatible with this eager native route. The path keeps BF16 arithmetic; it is
not yet an integrated, parity-validated or measured replacement.

The pinned native training implementation is named a prototype and currently
allocates/computes filter gradients even when filters are frozen; its FLA
`short_conv` shim targets one-token decode updates, not this full-sequence
Transformers training call. Neither the shim nor generic dense convolution is
a drop-in packed-training replacement. A scoped adapter to the native packed
operation and matched forward/input-gradient/adapter-gradient checks are needed.

The existing update-15 trace spends 70.483 ms in convolution forward and
143.403 ms in backward, approximately 4.31% of summed CUDA kernel time combined.
This supports a bounded follow-up, not a prediction of equivalent wall-time
savings or a claim NVIDIA is faster/more stable than the current Dao kernel.

## Completed NVIDIA BF16 causal Conv1D screen, 2026-10-05

Reject the native NVIDIA width-four SiLU convolution as a replacement in the
current resident FP4-MLP configuration. All twenty warmed finite updates complete,
but take 9.55% more time than the unchanged resident control and the matched
first-batch model loss/adapter-gradient check fails. Keep Dao convolution and
the existing FP4-MLP/BF16-GDN/FA4 timing baseline; preserve the general BF16
training default separately.

The scoped integration forwards existing CUDA int32 cumulative offsets through
the disabled GDN shell to `cudnn.ops.causal_conv1d`, preserving Accelerate hooks,
frozen parameters and packing boundaries. Every unsupported native route fails
closed. Scoped training/forward cache capacities become 256, avoiding upstream
64/128-plan eviction. Exact native sources, driver/integration sources and
checksum-bound receipts are archived per trial. The frontend/runtime and shared
compiler caches remain the previously pinned NVIDIA stack; no dependency upgrade
or worker restart occurs.

The native canary includes sequence lengths 1, 2, 3, 63, 65, 256 and 1025 with
8192 channels. Output/input-gradient relative L2 errors versus Dao are
0.004874%/0.002948%; versus independent FP32 convolution they are
0.16560%/0.16591%. Output and input-gradient cross-example leakage are exactly
zero. All compared values are finite. This component agreement does not establish
whole-model parity.

`06nvidiaconv` stops before updates: first-logical-batch loss changes from
0.5218598843 to 0.6128203273, absolute difference 0.0909604430, and adapter-gradient
relative L2 differs by 113.75%. `07convdiagnostic` reproduces these values exactly
and compares all 168 actual convolutions with Dao on their shared inputs. The
largest relative output difference is 0.012388%; the largest absolute difference
is 0.0625. `08convpassthrough` returns Dao outputs/gradients through the same
wrapper and exactly reproduces baseline loss and adapter gradients (zero relative
error). Its intentional diagnostic stop is recorded independently. This rules
out the wrapper as the source of the discrepancy and shows that small convolution
arithmetic differences trigger model-level divergence. It does not isolate the
amplification to FP4 MLPs rather than GDN recurrence or other downstream work.
No matched BF16-MLP diagnostic or candidate performance trace was run.

A separate, predeclared finite timing-only continuation retains the failed
5%/0.005 model gate and excludes quality/recipe selection. It reuses the isolated
canary and failed parity with matching worker/master/native sources, and verifies
that the native first preparation loss reproduces the recorded value. In
`09convfinite`, twenty batches prepare in 149.78847 seconds without optimizer
updates or master changes. Preparation builds 104 additional backward and 16
forward plans. Its first finite actual update then stops on an incorrect audit:
the first frozen GDN legitimately requires only forward execution. Seven native
inference plus 161 native autograd calls correctly cover the seven physical
rows. Retain the audit failure and baseline-restoration receipt.

`10convwarm` corrects the assertion and reuses the entire preparation receipt,
checking integration/native source identity, initial master, the same worker,
completed preparation and restored baseline. No preparation replay or model
reload occurs. Twenty updates reproduce the historical 147 physical partitions
from initial master
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
All updates have finite/nonmissing gradients and zero native forward/backward
compilations, native FP4 plan additions, Triton specializations, Dynamo graphs
or Inductor graph misses. The route audit counts 147 inference and 3381 autograd
calls: exactly 24 convolutions per row, with gradients where required.

| Condition | Mean seconds/update, updates 11–20 | Mean training loss, updates 1–20 |
| --- | ---: | ---: |
| Existing pooled Dao control | 3.67836 | 0.50287 |
| NVIDIA BF16 Conv1D | 4.02959 | 0.44356 |

All ten measured candidate updates are slower than their per-step pooled
controls; mean time increases by 9.54875%. Trial wall time, including reset and
adapter export but no preparation, is 94.62246 seconds. The matched mean absolute
training-loss difference is 0.08516; lower mean training loss and final loss
0.29021 do not establish held-out quality or convergence improvement.
Compilation during measurement is excluded as an explanation. Frozen-filter
gradient work in the current native backward remains a plausible cost, not a
measured causal attribution. Both timing and numerical evidence reject selection.

The final adapter has 256 FP32 tensors and 679,511,752 bytes. Adapter/receipt/source
archives and logs are collected locally; the baseline convolution methods and
cache settings are restored. PID 11905 remains alive and idle on the same B200
with its model, compiled kernels and packed-weight caches resident. No after-turn
heartbeat is promised because this session lacks a scheduling tool. Thirteen
new focused CPU tests and thirteen resident-worker checks pass, with Ruff and
whitespace checks. The exact executed integration predates subsequent type-hint
formatting, and the helper/shim archives predate their whitespace/reload cleanup;
arithmetic is unchanged. Retain the executed archives for provenance.

Completed receipt SHA-256:
`3c077f1e401043a93637005f215cc18940c8ceb3fe2b4d84674fb9d9c492603d`.
Reused validation SHA-256:
`b5480bc7fd09f7a2277fed275a5db20a304c82eb501f488c9d20d5b58d4dc091`.
Final adapter SHA-256:
`4e1ce98717dec1f878ece6a5f8e51245a401485ff6d1bbaba9fdae2403085ec1`;
final master tensor digest:
`881ec977ac54b538ac72f7488a176597368ee5b6ecc7ae9e968898c51933f3e6`.
The failed first-model parity reference SHA-256 is
`2efd07944f56dd4e41179ff43dd753ad52e90d112c89c70b114146213b2f7f8c`;
the isolated canary reference is
`5befbb5b152a8e5c75c88a5dcbba11836ba0572c6de802a22a5c55280adf0772`.

### CPU dispatch and normalization-copy screen

On 2026-10-05 the user requests investigating CPU dispatch and dtype conversions.
All screens reuse resident02 PID 11905, its model/shared caches and the existing
pooled 3.67835565-second FP4-MLP/BF16-GDN/FA4 control. No FA4/BF16 control or
unchanged startup probes are repeated. Each changed path receives one matched
first-logical-batch loss/gradient check; masters, RNG and physical partitions
remain fixed. Finite/missing-gradient checks stay enabled. Select only with at
least 2% less mean time on synchronized updates 11–20 and zero preparation in
those updates. Failed numerical gates stop before optimizer updates.

The CPU metadata intervention builds validated boundaries/positions/sequence IDs
and FlashQLA chunk metadata from known CPU lengths, using the pinned backend's
existing prepared-varlen hook. The FA4 router avoids repeated GPU boundary reads
only for an unchanged, shared Q/K offset tensor with matching operand shapes and
maximum lengths. Copied/mutated/untrusted offsets use original validation. The
nonblocking-input variant preserves Trainer recursive preparation and enqueues
input transfers on the original CUDA stream, without changing input storage.
Scoped contexts restore the original methods/router after every trial.

| Uninstrumented screen | Mean seconds/update, updates 11–20 | Time versus pooled control |
| --- | ---: | ---: |
| Existing resident02 FP4 control | 3.67836 | reference |
| `11dispatch`: CPU-prepared metadata | 3.97877 | +8.17% |
| `14async`: metadata plus nonblocking Trainer inputs | 3.79490 | +3.17% |
| `16asynconly`: nonblocking inputs, original metadata | 3.83429 | +4.24% |

None passes timing selection. All three uninstrumented trajectories and the
separate `13hotprofile` diagnostic complete twenty finite updates, preserve all
147 physical partitions and reproduce the control's loss/gradient/learning-rate
history and final FP32 adapter exactly. All measured updates add zero native
plans, Triton specializations, Dynamo graphs or Inductor graph-cache misses.
The metadata path's first targeted check takes 18.38420 seconds, including its
new graph guards; asynchronous validation takes 9.41920 seconds. `14async`
reuses checksum-bound `13hotprofile` validation in the same worker with matching
integration, installer, controls, initial masters and completed warm trajectory.
It performs no new numerical comparison or preparation. The isolated-transfer
check takes 9.08987 seconds. No complete twenty-batch preparation replay occurs.

`13hotprofile` records update 15 with the same shape-recording profiler as the
earlier resident01 trace. Stream synchronization calls fall from 268 to 59
(78.0% fewer): repeated compiled-region offset transfers, sequence-ID size reads
and FlashQLA metadata fallback synchronizations disappear. Remaining calls are
22 input/layout transfers, 36 boolean scalar reads and one other scalar read.
Their summed CPU wait durations are 714.66640 ms, versus 683.49770 ms in the
earlier trace; these waits overlap GPU execution and cannot be counted as
potential additive savings. Removing many calls mostly moves waits to later
operations. Kernel count falls only from 46,144 to 45,891, and summed GPU kernel
time changes from 4.96062 to 4.94416 seconds. The device span/gap are
6.15599/1.21773 seconds versus 5.96702/1.01226 previously. Instrumentation and
different resident-process preparation limit wall-time comparisons; counts and
operator ancestry establish what was removed, not an unprofiled speedup.

The old worker receipt incorrectly marks `13hotprofile` as uninstrumented
because the candidate, rather than the worker's profile variant, installs its
profiler. Preserve the raw receipt and the separate
`hotpath_screen_artifacts/instrumentation_correction.json`; exclude its
3.83235-second mean from speed selection. Future worker code recognizes the
exported trace when recording instrumentation. PID 11905 retains its original
loaded worker implementation; no restart occurs for that metadata correction.

Most explicit dtype traffic belongs to GDN Q/K normalization and its backward
path: the earlier trace has 1,034 BF16-to-FP32 rank-four/head-width-128 copies,
taking 199.83376 ms, and 1,034 reverse copies, taking 63.97676 ms. Input promotion
is only part of the first subtotal; backward also promotes incoming gradients.
Head replication/contiguity copies contribute another 127.06929 ms. Small
LoRA input casts are often already eliminated/fused by the compiled shells;
the trace does not support assigning all copy time to adapters or FP4 packing.

`12normcopy` removes the separate input promotion by calling the existing FLA
normalization with BF16 input and FP32 output. Normalization computation/output
and recurrent BF16 boundaries are intended to remain fixed, but the first
whole-model check fails: control loss 0.52185988 becomes 0.65588129 (absolute
difference 0.13402140), with adapter-gradient relative L2 1.03410778 (103.41%).
No optimizer updates or full preparation follow. `15normfixed` repeats only
the changed diagnostic while assigning all fifteen resident FP32 forward
normalization tilings to the BF16-input specialization. It reproduces exactly
the same failed loss/gradient values. Cache entries and baseline methods are
restored. Matching launch tiling is insufficient to repair this dtype change;
the source of numerical differences is not established. Keep both failed
receipts and the unchanged 5% gradient/0.005 loss gate. No finite-only timing
continuation or quality acceptance is asserted for either normalization path.

Retain the original resident FP4 baseline. This screen finds no verified
throughput improvement; fewer synchronization calls alone are insufficient,
and the tested normalization-copy removal does not meet model parity. Sixty-five
focused CPU checks pass, with Ruff and whitespace checks. Exact sources, receipts,
trace/analyzers, logs and checksum-verified adapter references are collected in
`results/b200_mlp_gemm/resident02/`. The four duplicate adapters match the
already collected control byte-for-byte, SHA-256
`0b9ea36eb19d012c730107ae222b490b4ad10620c795646d93cae8f94e68b71e`,
and retain 256 FP32 tensors. PID 11905 remains alive and idle with model/caches
resident. No after-turn heartbeat is promised; this session has no scheduler.

Diagnostic trace SHA-256:
`a58cbcef1c72925544a7049132a3b3275cc6d5b1bfe817b9517a7ae7a5c96728`.
Initial master remains
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`;
all completed trajectories finish at
`cf38e3e6881cebba402f3d16a1ed2d31dd3a259876375e887b2821cc9426fe3f`.

The collected campaign summary SHA-256 is
`e49e58e161df931f63420f7adf30ed45e71ce26f68a4183a4b1c096978b3c008`;
all 46 receipt/source files in its collection manifest are hash-verified.
`16asynconly/receipt.json` SHA-256 is
`39608ef24067258481c12c6c217818f51076dd4f8acf6e2e9fc6af1cee2e9216`.
Both failed normalization validation hashes are retained in that summary.

### Native grouped Q/K GDN screen

On 2026-10-05 the user authorizes testing removal of GDN Q/K head expansion.
The existing pinned FlashQLA SM100 kernel supports 16 shared Q/K heads and 32
value heads, including grouped-gradient reduction. A scoped source transform
removes only the two upstream `repeat_interleave` calls in all 24 GDN layers.
It preserves the Accelerate convolution hook, disabled GDN shell, frozen BF16
projections, BF16 recurrent operands, FP32 normalization/gates and original FP32
master adapters. Cached decoding is excluded. No FA4, MLP, objective, packing,
input-transfer or metadata implementation changes. The existing resident02
pooled 3.67835565-second control is reused; no fresh FA4 control runs.

`17grouped` checks one actual first logical batch before updates. Loss is
0.51918769 versus 0.52185988 (absolute difference 0.00267220, passing <=0.005),
but adapter-gradient relative L2 is 0.18044884 (18.0449%), failing the unchanged
5% gate. It stops with zero updates and restores the initial masters/baseline.
Its targeted check takes 148.65605 seconds, including the first grouped native
kernel compilations, in the shared persistent cache.

`18groupednorm` forces grouped forward normalization to use the resident launch
configuration for the corresponding expanded 32-head shape. It keeps FP32
input/compute/output and restores every modified cache entry after each call.
First-batch loss becomes exactly 0.52185988, but gradient relative L2 remains
0.17584170 (17.5842%), so this check also stops before updates. It takes
9.63274 seconds using existing native kernels. Matching forward reduction tiling
repairs this scalar loss difference but does not establish backward parity.

Source inspection identifies a changed numerical order: native grouped
backward sums per-value-head BF16 Q/K gradients with FP32 accumulation and
stores the sum in BF16 before FP32 normalization backward. With explicitly
repeated Q/K, normalization backward runs separately per value head before
the model sums gradients back into shared heads. This is a plausible source
of the remaining gradient difference; these checks do not prove it is the only
cause. No fused-gradient correction or new attention kernel is implemented.

Under the user's earlier instruction to measure finite variants regardless of
gradient differences, `19groupedfinite` continues solely for timing, retaining
both failed strict receipts. It binds the existing diagnostic to integration
source, installer AST, precision option, worker PID, initial master, matched
physical contract and restored baseline. It repeats no numerical comparison
or model-preparation replay. All twenty updates are finite, but measured
updates 12, 14 and 19 each add a Triton specialization: the raw 4.03276119-second
mean is not warmed and fails timing eligibility. Retain its completed receipt
and separate audit failure. The repeat `20groupedwarm` reuses the diagnostic
and all now-prepared shapes, resets the same adapters/AdamW/scheduler/RNG, and
performs only actual updates. The two trajectories have identical losses,
gradient-norm logs, partitions and final FP32 masters/adapters.

`21groupeddefault` restores default grouped normalization launch settings to
screen the original proposal independently of the forced diagnostic settings.
Its targeted comparison takes 10.02535 seconds and reproduces `17grouped` loss
and gradient differences exactly under the final integration source. The strict
result stays false, with separate finite timing-only acceptance. All twenty
actual updates are finite and fully warmed.

| Eligible uninstrumented condition | Mean seconds/update, updates 11–20 | Time versus pooled control |
| --- | ---: | ---: |
| Existing resident02 FP4-MLP/BF16-GDN/FA4 control | 3.67836 | reference |
| `20groupedwarm`: matched expanded normalization settings | 3.68516 | +0.185% |
| `21groupeddefault`: default grouped normalization settings | 3.70644 | +0.763% |

Neither meets the predeclared >=2% timing gain. Both are within 1% of the existing
control; these small differences do not establish a robust slowdown. Both
preserve all 147 physical partitions and add zero native MLP plans, Triton
specializations, Dynamo graphs or Inductor misses on every actual update.
Additional audits confirm no growth in five native TileLang caches: fused GDN
forward/backward, state preparation, KKT solve and grouped-gradient reduction.
No profiler is installed; no component-time attribution or removed-copy count
is inferred from these end-to-end measurements. Invocation wall times are
84.31330 seconds for the reused-check matched repeat and 95.40111 seconds for
the default-settings trial including its targeted check and final export.

Retain the original FP4-MLP/BF16-GDN/FA4 systems baseline and the general BF16
training recipe. Native grouping has no verified end-to-end throughput benefit
in this configuration and separately fails strict gradient agreement. Keep the
failed checks and un-warmed run rather than promoting on finite timing alone.
Forty-one focused CPU checks pass, with Ruff and whitespace checks. The same
worker PID 11905 remains idle and alive; forwards, normalization dispatch and
launch configuration entries are restored, and shared caches remain available.

Training loss is descriptive rather than a quality-selection metric here. Mean
logged loss over the twenty identical logical batches is 0.50287150 for the
existing control, 0.55002584 for matched grouped settings and 0.50216686 for
default grouped settings. Final logged losses are 0.39405805, 0.40921599 and
0.29194170, respectively. Changed gradients lead to different adapter
trajectories; these short training losses do not establish held-out quality.

All 56 receipt/source/log/metadata files in the collection manifest are verified
locally against remote hashes. Both unique saved adapters contain 256 FP32
tensors and 679,511,752 bytes. The first finite run's duplicate adapter is
represented locally by a checksum-verified link to the matched warmed repeat.
Artifacts are under `results/b200_mlp_gemm/resident02/grouped_screen_artifacts/`
and the respective trial directories. The campaign summary SHA-256 is
`80d704082bbad5049c023b72f735a5e593912ee263841c87cefc6390bf89281a`.
The matched/default warmed receipt hashes are
`b0c9b1698677837d63eadd667f45ab0426eacf5c1c77cb13ca02cc1f92623a02` /
`dabaa1bb60ab0e95a33e6f7da5bebcd68b7e189698bf9ced9be88ca9d54b93ee`.
Their adapter hashes are
`07ef92f75173cab99b342e6418212892da526b07189a1df7cc6c72c00d63cbd3` /
`dbc138a90c0a5ecc23cbdfbbb42c79f0aec9d737c64420a169cabbff27dae6d9`.
The original master identity remains
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`;
matched/default final masters are
`9f661c8173e876fa10bcf7e9a65032e59e63a3f01c1a1ac2b405cd2464d6c23f` /
`e93bf0d48991424c8be87909a4227587eff97f930fd10ba555b33f4fe37db32a`.

## Ordinary training recipe integration, 2026-10-05

The user explicitly selected the FP4 baseline as the default. The ordinary
Trainer now installs the same 32 native FP4 MLPs, hardware activation packing,
fused descale and physical-forward graph boundary through configuration. The
default profile aliases `qwen35_4b_b200_fp4_mlp`; the original BF16 comparison
is `qwen35_4b_b200_bf16_fa4`. No rejected GDN/convolution/dispatch/normalization/
grouped-head candidate enters the default. Shared launchers expose the pinned
cuDNN overlay and retain populated compiler/kernel caches. Master adapters stay
FP32 and original BF16 frozen parameters/state-dict layout stay intact.

Startup reuse binds the warmed03 metadata checksum and original kernel source
hashes, verifies the B200/FA4/cuDNN runtime and accepts the user's selected finite
policy for ordinary epochs or fixed step counts. Raw loss/strict-gradient
failures remain false; isolation/preflight remain separately recorded. The
experimental timing-only paths keep their twenty-update bound. Finite/missing
gradient checks during actual updates remain enabled. This integration does
not establish held-out quality equivalence or long-run convergence.

Validation: 121 focused CPU tests pass, including composed ordinary Trainer
commands, unsupported-recipe rejection, receipt/precision identity, master
parameter and graph-boundary preservation, historical failed receipts and
experimental/resident compatibility. Ruff and diff checks pass. A lightweight
smoke on the existing B200 verifies the ordinary command/config, real bound
receipt, cuDNN GEMM source/runtime and all 24 pinned FlashQLA bindings without
loading a model, running numerical probes or taking training updates. The
receipt is `results/b200_mlp_gemm/recipe_integration/runtime_smoke.json`.
Effective GPU compiler paths resolve the shared symlink to the populated
`student_injection_awareness/gpu-0` directory; cuDNN/CuTe caches stay under
`shared/` and FA4 retains its existing cache. Worker PID 11905 remains idle with
all model/native caches resident. No throughput control or new capacity is
launched. See [the decision](../decisions/b200_native_fp4_training_recipe.md).
