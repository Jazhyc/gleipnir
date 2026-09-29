# RTX PRO 6000 inference search — 2026-09-29

The frozen protocol and executable conditions are in
[`experiments/fp4_inference`](../../experiments/fp4_inference/README.md).
This campaign uses the existing ordered 32-row development slice (338,780 prompt
tokens) on Slurm job 32267015, with one RTX PRO 6000 Blackwell Server Edition,
one CPU and 32 GiB host RAM. GPU work stops at 01:29:21 CEST on 2026-09-30.

## Completed master/merge reference

The original four full prompts passed the unchanged FP32-master versus merged
BF16 eager gate on this GPU. Mean/max absolute score drift was 0.005537/0.017909,
correlation 0.999073, with zero threshold flips. The maximum base-to-master
adapter effect was 0.421917, establishing that the artifact changes outputs.
This is a bounded artifact gate, not a throughput or full-split quality result.

Transformers used SDPA and its Torch gated-delta fallback because the optional
fast libraries are unavailable. This exception is confined to twelve untimed
scores (base/master/merge on four rows); serving measurements use persistent
vLLM. The frozen serving inputs, tokenizer, parent subset and merge manifest
retain their original hashes. The first import waited on shared-filesystem I/O;
this startup delay is excluded from subsequent warmed throughput measurements.

Evidence: `results/local_inference/reference.json` and
`logs/slurm/fp4_inference/reference.log`.

The first serving attempt stopped before engine construction because the
historical benchmark expects its output parent directory to exist. The new
campaign launcher now creates that parent; the failed process-timing artifact
and execution receipt are preserved as `baseline_startup_failure` artifacts.
No GPU timing or prediction was produced by that attempt.

## Completed BF16 serving baseline

One ordered pass scored all 32 rows / 338,780 prompt tokens in **9.947606 s**,
or **34,056.44 prompt tokens/s**. Source-macro AUROC was 0.921488,
pAUROC@20 0.756198, Brier 0.098918; pooled AUROC 0.909804. Gloom/STRIDE
AUROC was 0.842975/1.0, with 30 distinct scores. These are development metrics.
All prediction identities, token totals, logit margins and logprob-to-score
normalizations were independently checked against the frozen JSONL.

The serving gate passed against both eager references. Versus the FP32 master,
mean/max score error was 0.004477/0.017909, correlation 0.998958, zero flips.
Versus eager merged BF16 it was 0.001060/0.004241, correlation 0.999952.
FlashAttention 2 and Triton/FLA GDN were selected. Engine construction took
210.770 s, canary/longest-input warmup 4.080 s, whole subprocess 247.430 s.
Cold compilation took 87.90 s, followed by 41.69 s initial profiling/warmup.
Do not count these costs as warmed scoring throughput.

The scoring telemetry sample showed 100% GPU use, 58 C, 2,377 MHz and
77,519 MiB total device memory; sampled software/hardware thermal slowdown was
inactive. The 0.80 utilization policy fills memory with KV capacity, so total
device usage is not model-weight memory. Ten-second telemetry is coarse for
this short pass. Workers exited successfully. Evidence:
`results/fp4_inference/baseline/` (including `prediction_audit.json`) and
`logs/slurm/fp4_inference/20260929T195517Z-benchmark.log`.

## Completed 4,096-token BF16 screen

Keeping two sequences and increasing only the prefill budget gave 9.751906 s,
34,739.87 tokens/s: **1.0201x**, below the >10% interest rule and inconclusive
with one pass. Canary parity passed. Paired full-split score drift had
mean/max 0.005816/0.031209, correlation 0.999545 and one threshold flip.
Macro AUROC/pAUROC were unchanged; Brier increased to 0.101238. Scores went
from 30 to 31 distinct values. The flip reduced macro FPR to 0.045455.
No scoring telemetry sample was captured for this condition, limiting thermal
comparison. Preserve the baseline; do not promote this small gain. Evidence:
`results/fp4_inference/bf16_b4096_s2/comparison.json` and its process artifacts.

## Completed 8,192-token / two-sequence BF16 screen

Increasing the budget to 8,192 at two sequences gave **9.545245 s**,
35,492.02 tokens/s, **1.0422x** versus the baseline. This remains below the
interest threshold. Serving parity passed; full-split mean/max score drift
was 0.003801/0.030490, correlation 0.999666, zero flips. Macro AUROC/pAUROC
were unchanged; Brier was 0.096458. Automatic telemetry captured 21 samples
across startup and scoring. No robust speed gain or quality improvement is
claimed from this single pass. Evidence: `results/fp4_inference/bf16_b8192_s2/`.

## Completed 8,192-token / eight-sequence BF16 screen

Increasing concurrency to eight at the same budget gave **9.561936 s**,
35,430.06 tokens/s, **1.0403x** versus baseline and slightly slower than the
two-sequence 8,192 condition. Canary parity passed. Paired mean/max score drift
was 0.002927/0.030967, correlation 0.999760 and zero flips. Macro AUROC and
pAUROC stayed unchanged; Brier was 0.099932, with 30 distinct scores.
The concurrency increase provides no demonstrated speed benefit. Prioritize
linear precision/kernel exploration while retaining both schedules as controls.
Evidence: `results/fp4_inference/bf16_b8192_s8/`.

## Completed disjoint activation capture

The existing capture recipe completed all twelve full trajectories, preserving
eight calibration/four held-out rows and 256 token positions at six projections
(layers 0,16,31). Scoring-split and serving-canary identity exclusion was checked.
The capture manifest records parent/input/merge hashes and checksums for each
activation file and the weights. It used bounded SDPA/Torch gated-delta fallback
hooks, not vLLM activations. It supplies native FP4 validation and reconstruction
data without modifying weights or observing scoring labels. Evidence:
`results/fp4_inference/capture/manifest.json` and the activation-capture log.

## Completed native CUTLASS FP4 canary and kernel screen

Native CUTLASS NVFP4 passed independent packed-value decoding plus FP32 GEMM
references on both real layer-0 shapes. Implementation relative L2 was
0.001659/0.001657 for gate/up and down, within the predeclared 0.01 bound.
Relative reconstruction error against original BF16 weights/activations was
**0.114780/0.105079**; this quantization error remains distinct from correct
native execution. No judge-score equivalence follows from these checks.

One 256-call window per condition, including activation amax reduction,
quantization/packing, allocation and GEMM, gave:

| Projection | BF16 ms | Complete native FP4 ms | Speedup |
| --- | ---: | ---: | ---: |
| Gate/up (2048,18432,2560) | 0.531873 | 0.179417 | 2.964x |
| Down (2048,2560,9216) | 0.253100 | 0.111806 | 2.264x |

Offline weight conversion was excluded; output allocation was included for
both precisions. Same-process timing, five warmups and three-second BF16
preconditioning were used. Before/after samples showed 54–64 C, 2,310–2,370 MHz
and no software thermal slowdown. No repeated-window variance is estimated.
This motivates a full-model MLP FP4 diagnostic while leaving BF16 as reference.
Evidence: `results/fp4_inference/kernel_cutlass/result.json` and kernel-cutlass log.

## Completed full-model MLP FP4 diagnostic

The native online CUTLASS method converted the 64 decoder MLP projections,
leaving attention/GDN projections, embeddings and the language head in BF16.
It loaded successfully, compiled, produced finite canaries/longest-input output,
and completed all 32 rows in **7.113271 s**, **47,626.47 tokens/s**, **1.3985x**.
Engine construction took 91.794 s; reported model allocation was 5.01 GiB.

The original serving gate **failed**: versus the master, mean/max canary
score error was 0.125850/0.255398, correlation 0.968362 and one flip. The
predeclared diagnostic override is recorded; failed parity was not relabeled.
Against the matched full-split BF16 outputs, mean/max drift was
0.024838/0.185333, correlation 0.992819 and two flips. Macro AUROC/pAUROC/Brier
were 0.929752/0.801653/0.096866, with 31 distinct scores. These small development
ranking changes do not establish a better monitor or justify promotion.

The scoring sample showed 54 C, 2,362 MHz and no software thermal slowdown.
This confirms an end-to-end speed opportunity, with inadequate score fidelity.
Keep BF16 as reference and explore selective precision, quantization scaling and
other native backends. The code uses native FP4 kernels, rejects emulation,
checks weight reconstruction, and records its additional source hashes in the
execution receipt. Decoder-layer selection, packing and deadline tests passed
(24 focused tests). Evidence: `results/fp4_inference/nvfp4_mlp_cutlass/` and
`logs/slurm/fp4_inference/20260929T201408Z-benchmark.log`.

## Completed per-channel FP8 diagnostic

Stock online FP8 with per-channel weights and per-token activations took
**6.959485 s**, **48,678.89 tokens/s**, **1.4294x**. It was slightly faster than
MLP-only FP4 in these single passes, because FP8 also accelerates other decoder
projections. The original canary failed its mean-error bound: master mean/max
0.027231/0.062176, correlation 0.992531 and one flip; merged mean 0.021694.
Full-split mean/max drift was 0.022805/0.117002, correlation 0.996916 and one
flip. Macro AUROC/pAUROC/Brier was 0.925620/0.776860/0.096712, with 31 distinct
scores. Retain this as a diagnostic tradeoff, not a parity-passing replacement.
Evidence: `results/fp4_inference/fp8_channel/`.

## Selective FP4 startup failure and cache identity fix

The first attempt keeping MLP layers 0 and 31 in BF16 failed before canaries
with `KeyError: weight_scale` while loading an AOT-compiled forward. The cache
key had omitted environment-based custom quantizer flags, so it reused the
all-MLP-FP4 graph for a different parameter layout. No throughput or quality
result exists for this failed attempt; all artifacts remain under
`nvfp4_mlp_keepends/`.

The launcher now resolves custom configurations with all quantizer flags and
the implementation SHA-256 in `engine.additional_config`, which vLLM includes
in its compile hash. A focused test verifies that selective precision changes
this identity without mutating the checked-in contract. Runtime configurations
and their source hash are retained. The retry uses a new `keepends_v2` output.

## Completed selective FP4 retry

The compile-identity fix allowed the mixed layout to load and score all rows.
Keeping MLP layers 0 and 31 BF16 gave **7.325875 s**, 46,244.31 tokens/s,
**1.3579x**. Master canary mean/max error fell to 0.081046/0.224908, but
correlation was 0.933810 with one flip: the original gate still failed.
Full-split mean/max drift was 0.026114/0.215126, correlation 0.990841 and one
flip. Macro AUROC/pAUROC/Brier: 0.925620/0.780992/0.094867. Keeping the two
end layers did not recover score fidelity or improve the full-split worst error.
No promotion. Evidence: `results/fp4_inference/nvfp4_mlp_keepends_v2/`.

## Completed native B12X kernel screen

FlashInfer B12X passed the same independent quantized FP32 arithmetic gate:
relative L2 0.001659/0.001657 for gate/up and down. Complete online calls took
**0.169318/0.115939 ms** versus BF16 0.531860/0.253152 ms, or
**3.141x/2.183x**. Original BF16 reconstruction errors remained 11.48%/10.51%.
Timing includes activation range reduction, packing, allocation and GEMM;
offline weight conversion is excluded. One 256-call window, 55–63 C and
2,302–2,362 MHz, no sampled software thermal slowdown. These measurements
do not establish a consistent B12X/CUTLASS winner; compare full-model serving.
Evidence: `results/fp4_inference/kernel_b12x/result.json`.

## Completed block FP8 startup failure

Online per-block FP8 selected `CutlassFp8BlockScaledMMKernel`, loaded weights
and compiled, then failed during engine profiling with
`cutlass_gemm_caller ... c3x/cutlass_gemm_caller.cuh:51, Invalid status`.
No canary or throughput result was produced. The failing matrix shape and
specific CUTLASS constraint have not been isolated. Retain all artifacts and
test the same recipe with the explicit Triton linear backend separately.
Evidence: `results/fp4_inference/fp8_block/` and
`logs/slurm/fp4_inference/20260929T202539Z-benchmark.log`.

## Completed full-model B12X FP4 diagnostic

The same MLP-only precision layout on FlashInfer B12X took **7.238406 s**,
46,803.12 tokens/s, **1.3743x**. It did not beat CUTLASS's 7.113271 s in these
single passes. Original canary parity failed: master mean/max error
0.106223/0.224908, correlation 0.964185, one flip. Full-split mean/max drift
was 0.026731/0.185333, correlation 0.992854 and two flips. Macro
AUROC/pAUROC/Brier: 0.915289/0.756612/0.099983. Backend rounding changed
some scores despite matching decoded arithmetic at the kernel canary.
Neither FP4 backend is a fidelity-qualified replacement. Evidence:
`results/fp4_inference/nvfp4_mlp_b12x/`.

## Completed Triton block FP8 recovery

Changing only the block FP8 linear backend to Triton recovered startup and
completed all 32 rows in **8.784008 s**, 38,567.81 tokens/s, **1.1325x**.
The original canary passed: master mean/max 0.009938/0.034378, correlation
0.996331 and zero flips; merged mean/max 0.004400/0.016469.
Full-split mean/max drift was 0.013666/0.113118, correlation 0.997551 and one
threshold flip. Macro AUROC/pAUROC remained 0.921488/0.756198; Brier
was 0.102096 (+0.003178), within the frozen screening bound.

This is the first candidate to clear both the >10% speed interest rule and
the original canary/development quality screens. One pass does not establish
repeatability, population quality, or full-split score equivalence; its largest
development score difference exceeds 0.1 despite passing the four-row gate.
Retain BF16 and confirm this finalist with a matched repeat. Evidence:
`results/fp4_inference/fp8_block_triton/`.

## Completed disjoint NVFP4 clipping screen

A new Triton packer passed independent nearest-even E2M1 tie/sign checks,
exact padded scale swizzle checks, and native GEMM versus decoded FP32 checks.
The largest native implementation error across all recipes/projections/splits
was 0.001675. At clip 1, decoded weights differed from stock CUDA packing by
at most 0.007863 relative L2, within the 0.01 predeclared check. It is an
independently validated alternative, not a claim of bitwise stock equivalence.

The frozen 72-recipe screen selected **unclipped dynamic scaling** (weight and
activation clip 1.0). Mean calibration output relative L2 across six projections
was 0.094842; the same selected recipe gave held-out 0.095005. Rounded-up
power-of-two scaling at clip 1 was 0.094894 on calibration. Group clipping did
not improve the objective, so no clipped recipe is promoted. Held-out errors
never selected the recipe; no serving scores or evaluation labels were used.
The grid completed in 12.75 s after loading/JIT. Next compare complete kernel
costs for the unclipped Triton packer, preserving the serving fidelity gate.
Evidence: `results/fp4_inference/quantizer_screen/result.json`.

## Completed MLP-only per-channel FP8 screen

Leaving attention/GDN projections BF16 and using FP8 only in MLPs gave
**8.049067 s**, 42,089.35 tokens/s, **1.2359x**. Both original canaries passed:
master mean/max error 0.002004/0.008015, correlation 0.999835, zero flips;
merged mean/max 0.005421/0.017909. Full-split mean/max drift was
0.012884/0.089178, correlation 0.998186 and one flip. Macro AUROC/pAUROC
stayed 0.921488/0.756198; Brier was 0.096286. This is a faster initial
fidelity-qualified candidate than full-decoder Triton block FP8. Selective
precision recovered the canary relative to all-decoder per-channel FP8, with
less speed gain. Confirm this finalist with matched repeats. Evidence:
`results/fp4_inference/fp8_mlp/`.

## Completed alternative FP4 packer kernel screen

The unclipped Triton packer with native CUTLASS GEMM passed independent decoded
FP32 references (relative L2 0.001659/0.001657). Complete gate/up and down calls
took **0.182475/0.122285 ms**, versus BF16 0.531925/0.253392 ms:
**2.915x/2.072x**. Original BF16 reconstruction was 0.114778/0.105079.
This did not improve on the stock CUDA packer screen (0.179417/0.111806 ms).
One timing window per path does not resolve small differences, but there is
no measured reason to promote the new packer for speed. It remains a validated
research primitive for altered quantization. Evidence:
`results/fp4_inference/kernel_triton_pack/result.json`.

## Completed FlashInfer BF16 attention screen

Replacing FlashAttention with FlashInfer attention at the baseline schedule
gave **9.928109 s**, 34,123.32 tokens/s, **1.0020x**. Both original canaries
passed; full-split mean/max drift was 0.004111/0.030490, correlation 0.999702
and zero flips. Macro AUROC/pAUROC stayed unchanged, Brier was 0.098282.
This does not clear the speed interest rule and supplies no measured advantage
over the default attention backend. Evidence:
`results/fp4_inference/bf16_flashinfer/`.

## Completed Blackwell bounded GPU profiles

Early CUPTI probes succeeded and the identical sixteen-step diagnostic captures
contained 13,328 BF16 and 13,456 MLP-FP8 CUDA kernel events. By heuristic kernel
name grouping, summed durations were:

| Category | BF16 share | MLP FP8 share |
| --- | ---: | ---: |
| Linear GEMMs | 70.94% | 63.81% |
| Attention | 11.38% | 14.16% |
| GDN and convolution | 9.85% | 11.92% |
| Fused elementwise/reduction | 5.83% | 7.60% |
| Other | 1.99% | 2.50% |

Duration sums were 936.858/752.162 ms. These are instrumented partial traces,
not serving elapsed times or an end-to-end speed estimate. Template type names
inside attention kernels are not counted as GEMMs; a focused classifier test
guards this. Linears remain the primary target, motivating mixed channel/block
FP8 outside MLPs while retaining exploration of attention and GDN kernels.
Evidence: `profile_bf16/` and `profile_fp8_mlp/` under campaign results, with
checksum-recorded traces and `kernel_summary.json`.

## Completed down-only FP4 diagnostic

Keeping gate/up BF16 and quantizing only down projections took **9.047070 s**,
37,446.38 tokens/s, **1.0995x**, just below the frozen interest threshold.
Canary parity failed: master mean/max error 0.046753/0.124353, correlation
0.999482, one flip. Full-split mean/max drift was 0.018998/0.092667,
correlation 0.997239 and three flips. Macro AUROC/pAUROC/Brier:
0.933884/0.801653/0.096522. Isolating down projections improved the worst
drift relative to all-MLP FP4 but still misses fidelity, with insufficient speed.
Evidence: `results/fp4_inference/nvfp4_down_cutlass/`.

## Completed gate/up-only FP4 diagnostic

Keeping down projections BF16 and using FP4 gate/up gave **7.852298 s**,
43,144.06 tokens/s, **1.2668x**. Original canary parity failed: master
mean/max error 0.057448/0.117002, correlation 0.880189, zero flips. Full-split
mean/max drift was 0.024930/0.168882, correlation 0.991734 and two flips.
Macro AUROC/pAUROC/Brier: 0.931818/0.777273/0.090961. Neither isolated
projection clears fidelity, even though gate/up accounts for more speed gain.
Retain both diagnostics; do not infer a safe FP4 projection from ranking gains.
Evidence: `results/fp4_inference/nvfp4_gate_up_cutlass/`.

## Completed mixed channel/block FP8 diagnostic

Per-channel MLP FP8 plus explicitly forced Triton block FP8 in the remaining
decoder linears loaded successfully and took **7.554153 s**, 44,846.85 tokens/s,
**1.3168x**. Original canary parity failed: master mean/max
0.043870/0.117002, correlation 0.999980, zero flips. Full-split mean/max drift
was 0.018962/0.086512, correlation 0.997103 and two flips. Macro
AUROC/pAUROC/Brier: 0.929752/0.780992/0.102662. Combining two separately
passing precision recipes does not preserve their canary behavior; retain MLP
FP8 as the stronger fidelity candidate. Component scopes remain frozen probes.
Evidence: `results/fp4_inference/mixed_fp8/`.

## Completed FP4-down / FP8-gate hybrid diagnostic

Using FP4 down and per-channel FP8 gate/up gave **7.782110 s**,
43,533.18 tokens/s, **1.2783x**. The original master canary mean error
improved to 0.028193 but still exceeded 0.02; maximum 0.093386, correlation
0.999351, one flip. Full-split mean/max drift was 0.013864/0.062419,
correlation 0.998326 and two flips. Macro AUROC/pAUROC/Brier:
0.919421/0.756612/0.097598. This is the best full-split fidelity among tested
FP4 layouts so far, but still fails the original gate and is not promoted.
Evidence: `results/fp4_inference/nvfp4_down_fp8_gate/`.

## Completed six-projection Hessian-feedback screen

The Triton feedback implementation passed exact packed-code/scale comparison
with an independent small NumPy reference. All native GEMM checks against
decoded FP32 references passed (maximum relative L2 0.001666).
Fixed-order, damping-0.01, 128-column GPTQ-style weight feedback reduced mean
W4A4 calibration reconstruction error from **0.094842 to 0.069032** (27.2%).
Held-out mean fell from **0.095005 to 0.090320** (4.9%). Gains were much smaller
on held-out inputs and two down projections worsened slightly, so calibration
error is not a population or judge-quality claim.

The fixed expansion gate passed: >=10% calibration improvement and <=2%
held-out worsening. Proceed to all decoder MLP projections on the same twelve
disjoint trajectories, with eight calibration rows only for fitting, then freeze
the artifact before the unchanged serving gates. No hyperparameter adjustment
or selection used evaluation labels. This adapts second-order weight feedback
from [GPTQ](https://arxiv.org/abs/2210.17323) to native NVFP4 groups, rather
than claiming original GPTQ benchmark results. Evidence:
`results/fp4_inference/gptq_screen/result.json`.

## Completed attention-scope mixed FP8 screen

MLP per-channel FP8 plus attention-projection Triton block FP8, with GDN
projections BF16, took **7.898270 s**, 42,892.94 tokens/s, **1.2595x**.
Both original canaries passed: master mean/max 0.014942/0.030490,
correlation 0.999746, zero flips. Full-split mean/max drift was
0.013984/0.113118, correlation 0.998037 and one flip. Macro AUROC/pAUROC
stayed 0.921488/0.756198; Brier was 0.103558 (+0.004640), within the screen.
Its gain over MLP-only FP8 is small and unconfirmed, so retain both candidates.
Evidence: `results/fp4_inference/mixed_fp8_attn/`.

## Completed all-layer disjoint capture

The same twelve complete trajectories / 34,068 prompt tokens were captured
at all 32 MLP layers, adding coverage without changing IDs, split assignment,
uniform positions, input/merge hashes or prompt lengths. The twelve bounded
forwards took 6.496 s excluding loading and artifact writes. Transformers
SDPA/Torch fallback remains a hook-only exception. All 64 projection keys
and source hashes are retained. Original six captured weight matrices must
match exactly before export fitting proceeds. Evidence:
`results/fp4_inference/capture_all/manifest.json`.

## Completed all-MLP Hessian-feedback export

The fixed recipe fitted all 64 projections on the eight calibration trajectories.
Every native decoded-reference check passed (maximum relative L2 0.001858),
and sample original-weight reconstruction stayed below 0.25 (maximum 0.231951).
Mean output reconstruction across all projections was 0.077270 calibration
versus 0.102604 held-out. This larger gap limits generalization claims; no score
fidelity is established by the export. Summed per-projection fit/audit/write
work was 8.877 s after input loading.

The frozen manifest SHA-256 is
`9209b3da693baeb31ec15e3a6133361373d731d726da636dfcdecc6593484ab7`.
It binds the merge, calibration, algorithm and all packed files. The serving
loader requires that manifest hash, per-file hash, exact original BF16 weight
bytes, projection coverage, shapes/dtypes and finite positive scales before
using any fitted tensor. The FP32 master and BF16 merge are unchanged. Fitted
weights remain ignored. Evidence: `results/fp4_inference/gptq_all/manifest.json`.

## Completed fitted FP4 serving diagnostics

Prepared weights with Triton activation packing took **7.191664 s**,
47,107.32 tokens/s, **1.3832x**. The original master canary passed
(mean/max 0.015244/0.030490, correlation 0.996503, zero flips), but the merged
canary mean **0.020782** exceeded the unchanged 0.02 bound. Its maximum was
0.037297 and correlation 0.992028. The combined gate remains **failed**.
Full-split mean/max drift was 0.021444/0.086512, correlation 0.996980 and
one flip. Macro AUROC/pAUROC/Brier: 0.923554/0.766529/0.106803.
This substantially improved canary fidelity relative to naive FP4, with no
promotion because the complete original gate did not pass.

The identical prepared weights with stock CUDA activation packing took
**7.064651 s**, 47,954.24 tokens/s, **1.4081x**, but fidelity worsened:
master mean/max 0.036185/0.143609, correlation 0.978087; merged mean 0.041722.
Full-split mean/max 0.027818/0.123876, three flips. Macro AUROC/pAUROC/Brier:
0.907025/0.710744/0.115066, outside both development quality bounds.
Independently correct native arithmetic does not make these packers identical;
small packing differences propagate through the model. Evidence:
`results/fp4_inference/gptq_nvfp4_triton/` and `gptq_nvfp4_cuda/`.

## Completed fitted FP4-down / FP8-gate hybrid

Prepared FP4 down projections plus per-channel FP8 gate/up completed in
**7.682589 s**, 44,097.11 tokens/s, **1.2948x**. Both original canaries passed:
master mean/max 0.019111/0.034378, correlation 0.992937, zero flips; merged
mean/max 0.015694/0.030967, correlation 0.996787. The master mean is close
to its fixed 0.02 limit, so confirmation matters.

Full-split mean/max drift was 0.018450/0.061457, correlation 0.997443 and
**zero flips**. Macro AUROC/pAUROC/Brier: 0.919421/0.745868/0.102922.
AUROC loss 0.002066 and Brier increase 0.004004 satisfy the frozen development
screen; pAUROC declined 0.010331 and remains reported. This is the first
native FP4-containing condition to clear all declared selection gates. It is
a finalist, not a repeatability or population-equivalence result. Evidence:
`results/fp4_inference/gptq_nvfp4_down_fp8_gate/`.

## Completed reverse fitted hybrid diagnostic

Prepared FP4 gate/up with per-channel FP8 down took **7.353427 s**, **1.3528x**.
The master canary passed (mean/max 0.010069/0.040275, correlation 0.994847),
but the merged correlation **0.989657** failed the unchanged 0.99 requirement.
Merged mean/max were 0.015606/0.058184. Full development drift was mean/max
0.019600/0.086512, correlation 0.997257, zero flips. Macro AUROC/pAUROC/Brier
were 0.925620/0.780992/0.096612. This faster layout remains a failed diagnostic;
its development metrics do not override the canary. Evidence:
`results/fp4_inference/gptq_nvfp4_gate_up_fp8_down/`.

## FP32 output projection mechanism audit and interrupted session

The direct native GPU projection canary returned finite FP32 output and
relative L2 0.000003238/0.000002826 against independent FP32 references on
real captured inputs. Evidence: `results/fp4_inference/logits_canary/result.json`.
However, Qwen3.5 ties its output head to `VocabParallelEmbedding`; selecting
only `ParallelLMHead` did not install the method. The first completed full-model
controls therefore do **not** measure the intended FP32-head intervention.
Keep their artifacts and exclude them from selection under that interpretation:
`bf16_fp32_logits_v2` took 9.851273 s and passed the ordinary score gates;
`gptq_nvfp4_fp32_logits` took 7.183757 s and failed them. Their coarse raw
margins and the locked model source exposed the routing error.

The earlier `bf16_fp32_logits` process ended when its interactive tool session
disappeared, before model startup and result creation. The saved interruption
receipt records the absent worker and idle GPU. Subsequent runs use detached
`srun` steps inside the same allocation, with logs and deadline enforcement.
This protects runtime across tool-session refreshes; monitoring still requires
the active agent because no in-chat scheduling tool is available.

The corrected selector covers tied embeddings and untied heads. Inherited
embedding lookup stays unchanged; a focused test checks it. Actual full-vocabulary
projection execution must log its output shape and `torch.float32` dtype before
interpreting the corrected matched pair. The runtime deadline now allows 20 s
for process termination before the ten-minute allocation reserve begins.

## Kernel routing evidence
The installed locked vLLM 0.24.0 source has ModelOpt/compressed-tensors NVFP4
linear methods and FlashInfer B12X, CUTLASS and other NVFP4 kernel adapters,
but its online quantization registry has FP8/MXFP8 and no NVFP4 shorthand.
Current upstream documentation describes a newer online NVFP4 path, so local
source and executed routing govern this pinned campaign. Native SM120 support
must be established by execution before interpreting an FP4 speed result.
The [FlashInfer FP4 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
documents backend-specific layouts and B12X selection on SM120; its online
documentation is newer than the pinned package and is advisory.
