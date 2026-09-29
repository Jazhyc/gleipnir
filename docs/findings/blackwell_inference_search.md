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

The constructor-only retry `bf16_fp32_logits_v3` also lacked execution proof:
the locked Qwen3.5 constructor does not pass `quant_config` to its embedding.
It completed as another invalid mechanism control. The next retry uses a checked
post-load hook before warmup, preserving the original weight object and tied
embedding alias. Its success still depends on the actual FP32 execution log.

`bf16_fp32_logits_v3` took 9.852743 s with coarse margins and is excluded as
an intervention measurement. The post-load `bf16_fp32_logits_v4` took 9.924612 s
and passed both score gates (master mean/max 0.006469/0.011163; merged
0.004654/0.011154). Its fractional margins show the projection changed, but the
explicit execution log was silent because `init_logger(__name__)` was outside
vLLM's configured namespace. This is a logging failure, not evidence that the
worker lacked the flag. The next matched runs use a logger under `vllm` and
read the transferred quantization config rather than depending on environment
propagation. Both paths and the tied weight/lookup identity have focused tests.

The config-based post-load `bf16_fp32_logits_v5` passed both original canaries:
master mean/max 0.006465/0.013253, correlation 0.999726; merged mean/max
0.003620/0.009085, correlation 0.999804. It took **9.901755 s** (1.0046x),
with full-development mean/max drift 0.004185/0.019510 and one threshold flip.
Distinct scores increased from 30 to 32. Macro AUROC/pAUROC stayed
0.921488/0.756198; Brier was 0.098538. Fractional margins are consistent with
the native FP32 method; visible installation/dtype proof is required on the next
matched run and three-pass control. Evidence: `bf16_fp32_logits_v5/`.

## Completed fitted FP4 with verified FP32 full-vocabulary logits

The checked post-load hook installed the tied head, and actual native projection
execution logged shape `(2, 248320)` and `torch.float32` before scoring.
Fitted FP4 in all MLP projections with Triton activation packing took
**7.240453 s**, **1.3739x** versus the original BF16 baseline and **1.3676x**
versus the FP32-head BF16 control. It failed both original canaries: master
mean/max 0.050242/0.178143, correlation 0.937434; merged mean/max
0.055780/0.178143, correlation 0.921582. This is a verified negative intervention,
not another routing failure.

Versus the matched head control, full-development mean/max drift was
0.032917/0.232513, correlation 0.990350, one threshold flip and 32 distinct
scores. Macro AUROC/pAUROC/Brier were 0.904959/0.714876/0.104921: AUROC loss
0.016529 exceeded 0.01. Removing the coarse head rounding and disabling reduced
intermediate reductions did not rescue this FP4 layout. Both components changed
together in the frozen pair, so this result does not isolate their causal effects.
Keep the passing fitted-down hybrid as a finalist. Evidence:
`gptq_nvfp4_fp32_logits_v4/`, including separate baseline and head comparisons.

## Completed Marlin FP4 weight-only kernel screen

Native Marlin W4A16 passed all six independently decoded weight-only FP32
arithmetic checks (relative L2 0.001602–0.001660 < 0.005). On the real gate/up
shape, speedups over BF16 at M=1/128/2048 were **1.8224x/1.0636x/0.9165x**;
on the down shape, **1.0878x/0.8765x/0.9077x**. This path has a useful small-M
tradeoff but is slower than BF16 on both large prefill shapes in these windows.
It does not justify replacing native W4A4 in this long-prompt workload.

Each measurement used 256 complete calls with online padding/allocation/GEMM
included, five warmups and matched three-second BF16 heating. Native preparation
and weight repacking were offline. Temperature and clocks were recorded around
each window, with no observed thermal slowdown in those samples. These are
single kernel windows, not a serving-quality result or a decode throughput
benchmark. Evidence: `results/fp4_inference/marlin/result.json`.

## Completed three-pass BF16 confirmation

The second independent BF16 engine passed the original canaries. Its three
warmed passes were **9.936403/9.950902/9.965627 s** (median **9.950902 s**),
matching the initial 9.947606 s baseline. Scores and raw margins were identical
within this engine, with zero score range or threshold instability.

Across the independent starts, mean/max score drift was **0.004611/0.030967**,
correlation 0.999683 and zero flips. Macro AUROC/pAUROC/Brier on the second
start were 0.925620/0.776860/0.099289. This shows that zero within-engine repeat
noise does not establish identical outputs after restarting vLLM. Candidate
reports must preserve both the initial baseline and the matched repeated
baseline rather than treating changed rankings as a training improvement.
Evidence: `results/fp4_inference/bf16_confirm/`, including `independent_start.json`.

## Completed three-pass MLP FP8 confirmation

MLP-only per-channel FP8 passed both original canaries again. Warmed times were
**8.046348/8.051037/8.058522 s**, median **8.051037 s**, **1.2360x** versus
the matched three-pass BF16 median. Scores and margins were identical across
all three passes, and scores exactly matched its first independent engine.
Master canary mean/max stayed 0.002004/0.008015; merged 0.005421/0.017909.

Macro AUROC/pAUROC/Brier were 0.921488/0.756198/0.096286. Versus the repeated
BF16 baseline, AUROC loss 0.004132 and Brier change -0.003002 pass the fixed
development screen; pAUROC change -0.020661 remains reported. These relative
differences partly reflect BF16's between-start drift, and are not claims of
training gains. Evidence: `fp8_mlp_confirm/repeat_comparison.json` and
`independent_start_drift.json`.

## Completed three-pass fitted FP4 hybrid confirmation: rejected

Fitted FP4 down plus FP8 gate/up took **7.765169/7.769123/7.773791 s**,
median **7.769123 s**, **1.2808x** versus repeated BF16. Scores and margins
were identical within its three-pass engine. However, the second independent
start **failed** the original master canary: mean **0.022632 > 0.02** and
correlation **0.987979 < 0.99**, maximum 0.049461. The merged canary passed
(mean/max 0.019215/0.031552, correlation 0.993386). The initial pass's borderline
master mean 0.019111 did not provide enough margin to survive restart.

The development quality screen alone passed (macro AUROC loss 0.004132 and
Brier increase 0.007172 against repeated BF16), but the complete selection
gate is failed. Across hybrid engine starts, mean/max score drift was
0.015497/0.091250 and one threshold flip. Reject this layout as a confirmed
serving candidate; zero within-engine noise cannot override between-start
fidelity failure. MLP FP8 remains the confirmed passing candidate. Evidence:
`gptq_down_fp8_confirm/repeat_comparison.json` and `independent_start_drift.json`.

## Completed FlashInfer attention with FP8 KV cache

BF16 weights with FlashInfer attention and E4M3 KV storage passed both original
canaries (master mean/max 0.013160/0.030490, correlation 0.999898; merged
mean/max 0.007623/0.030490, correlation 0.999578). However, warmed scoring took
**10.530073 s**, **0.9447x** the baseline: it was slower. Full-development
mean/max drift was 0.006210/0.030967, correlation 0.999520 and zero flips.
Macro AUROC/pAUROC/Brier were 0.925620/0.757025/0.095181. This path does not
meet the >10% speed screen.

The requested `calculate_kv_scales=true` was explicitly disabled by locked
vLLM for the recurrent hybrid model because dummy-profile recurrent states give
unreliable scales. Actual scale behavior was the default 1.0, so this is a
**unit-scale FP8 KV** measurement, not a calibrated-cache result. The actual
BF16-query/E4M3-KV head-256 FlashInfer CUDA specialization compiled on the
allocated CPU, with `MAX_JOBS=1`; initialization took 182.317 s. Its real
one-token outputs and score gates passed after compilation. No cache-precision
deployment recommendation follows from memory-format support alone. Evidence:
`bf16_flashinfer_fp8kv/` and `20260929T220625Z-benchmark.log`.

## Completed larger-budget MLP FP8 screen

Keeping MLP per-channel FP8, two sequences and the original scoring contract,
raising the prefill budget from 2,048 to 8,192 took **7.818977 s**, **1.2722x**
versus the initial BF16 baseline and 1.0294x versus initial MLP FP8. Both
original canaries passed. Development mean/max score drift was
0.008938/0.062177, correlation 0.999234 and one threshold flip. Macro
AUROC/pAUROC/Brier were 0.921488/0.756198/0.100029; AUROC was unchanged and
Brier increased 0.001111. There were 31 distinct scores. This passes the
initial interest and quality screen, but requires a separate three-pass and
independent-start confirmation before selection. Evidence:
`results/fp4_inference/fp8_mlp_b8192_s2/`.

## Completed Triton activation-packer fitted hybrid screen

Holding fitted FP4 down weights, FP8 gate/up and the original scheduler fixed,
switching CUDA to the independently validated Triton activation packer took
**7.782019 s**, **1.2783x**. Both original canaries passed: master mean/max
0.011119/0.034378, correlation 0.995087; merged mean/max 0.007702/0.016469,
correlation 0.998028. Development mean/max drift was 0.017702/0.215123,
correlation 0.993984 and zero threshold flips. Macro AUROC/pAUROC/Brier:
0.921488/0.756198/0.093238. The development screen passed, although one
individual score moved substantially. Freeze a separate three-pass restart
confirmation; this initial result does not establish that the packer fixes the
previous CUDA hybrid's between-start failure. Evidence:
`results/fp4_inference/gptq_down_fp8_triton/`.

## Completed synchronous BF16 engine screen

Disabling v1 engine multiprocessing passed both original canaries and produced
development scores exactly matching the earlier three-pass BF16 engine. It
took **10.209538 s**, **0.9743x** the initial baseline; asynchronous scheduling
was still reported by vLLM. This flag changes the offline engine process path,
not the model's numerical precision or an established batch-invariance property.
The unchanged scores are one between-start observation, not a reproducibility
guarantee. No speed benefit was demonstrated. Evidence: `bf16_no_mp/`.

## Completed synchronous MLP FP8 engine screen

The matching MLP FP8 condition passed original canaries and exactly reproduced
the initial MLP FP8 development scores, but took **8.308580 s**, compared with
the asynchronous path's 8.049067 s initial pass and 8.051037 s repeated median.
Its 1.1973x speedup against BF16 is lower than the confirmed asynchronous FP8
gain. Both synchronous screens preserve scores observed in earlier engines;
neither demonstrates that process topology resolves numerical variability.
Retain the default process path. Evidence: `fp8_mlp_no_mp/`.

## Fused SiLU/FP8 kernel validation failure and compiler evidence

The bounded hand-written Triton SiLU/multiply plus per-token E4M3 packer
failed its first layer-0/M=128 stock-path arithmetic check: decoded-activation
relative L2 **0.011620 > 0.005**. Native CUTLASS multiplication of its quantized
outputs was correct against independent FP32 decoding (relative L2 0.001655),
so the failure precedes GEMM. Stop before all timing and serving integration,
as declared; this does not establish a speed result or a specific rounding cause.
The original partial result, failure receipt and exact source snapshots are in
`results/fp4_inference/silu_kernel/`, with the Slurm failure in its driver log.

The earlier real vLLM MLP FP8 profile already contains
`triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_max_mul_reciprocal_silu_slice_unsqueeze_4`
(384 calls, 14.012 ms summed GPU time). Thus the installed compiler already
fuses this SiLU/FP8 activation path. A stock two-CUDA-kernel comparison would
not by itself demonstrate an improvement over the compiled serving baseline.
Separately, the locked RMSNorm/quant matcher rejects the model's FP32 Gemma
norm weights paired with BF16 inputs; enabling its flag alone is not evidence
of executed fusion. Keep the existing compiled serving path.

## Batch-invariance startup rejection on Qwen3.5 GDN

The locked `VLLM_BATCH_INVARIANT=1` BF16 preflight loaded weights, then failed
before warmup or scoring with **`VLLM batch_invariant mode is not supported for
GDN_ATTN.`** The beta mode cannot represent this model's gated-delta attention
in this version. Preserve the 34.861 s failed startup and its logs; there is no
serving timing, score result or batch-invariance claim. Do not bypass the
backend's explicit support check. Evidence: `bf16_batch_invariant/`, its
execution receipt, and `20260929T222249Z-benchmark.log`.

## Triton-packed FP4 hybrid three-pass confirmation: rejected

The fresh-engine passes took **7.787093/7.793463/7.797013 s**, median
**7.793463 s**, **1.2768x** repeated BF16. Within-engine scores and margins
were identical, but the original master canary failed: mean **0.021178 > 0.02**,
maximum 0.049461 and correlation 0.993994. The merged gate passed (mean/max
0.017761/0.031552, correlation 0.997485). Thus changing activation packing did
not establish a restart-stable FP4 layout. Reject it as a confirmed candidate.

The development quality screen alone passed (AUROC loss 0.002066, Brier
change -0.000073 versus repeated BF16). Across the two Triton-hybrid starts,
mean/max score drift was 0.015649/0.246090, correlation 0.993181 and zero flips.
Neither favorable aggregate metrics nor zero within-engine noise override the
canary failure. Evidence: `gptq_down_fp8_triton_confirm/`.

## Completed larger-budget MLP FP8 three-pass confirmation

The fresh 8,192/two MLP FP8 engine passed both original canaries. Warmed times
were **7.831891/7.847283/7.869111 s**, median **7.847283 s**, **1.2681x**
against repeated BF16. The original 2,048/two FP8 median was 8.051037 s,
so the matched-budget gain is modest (1.0260x). Macro AUROC loss 0.004132 and
Brier increase 0.000741 against repeated BF16 pass the fixed development screen;
pAUROC change -0.020661 remains reported.

Unlike the original FP8 layout's exact repeats, one low-probability row varied:
maximum score range **0.002980**, raw-margin range 0.125, no threshold-unstable
rows. Between initial and repeated 8,192 engines, mean/max score drift was
0.000093/0.002980, correlation 0.999999 and no flips. This is small measured
variation, not an exact reproducibility claim. Evidence:
`fp8_mlp_b8192_confirm/repeat_comparison.json` and `independent_start_drift.json`.

## Calibration-selected FP4 down precision screen: rejected

The loaded-weight audit proved exactly 24 packed-uint8 native FP4 down
projections and eight E4M3 FP8 down projections, with the declared calibration
ranking and all gate/up projections FP8. Warmed scoring took **7.963925 s**.
Both original canaries failed: master mean/max 0.026663/0.062176, correlation
0.990514; merged mean/max 0.023246/0.062176, correlation 0.995258. Each had
one threshold flip. The development macro AUROC **0.911157** loses 0.010331
against the initial baseline, exceeding 0.01; Brier was 0.102231. Reject this
layout without further confirmation.

Protecting locally high-reconstruction-error layers did not recover final-score
fidelity. The activation screen's calibration-only ranking is not a sufficient
proxy for monitor-score sensitivity or accumulated hybrid-model errors. This
negative strengthens the case for testing an end-to-end teacher-distribution
objective in a separately authorized QAD pilot, without establishing that
training would succeed. Evidence: `gptq_selective_down8/` and the explicit
loaded-precision audit in `20260929T222749Z-benchmark.log`.

## Completed eight-sequence MLP FP8 scheduler screen

Keeping the 8,192-token budget and MLP FP8, eight sequences took **7.798921 s**,
**1.2755x** the initial baseline. Both original canaries passed (master mean/max
0.005537/0.017909; merged effectively identical to the eager merged reference).
Development mean/max drift was 0.015356/0.143609, correlation 0.997032 and one
threshold flip. Macro AUROC/pAUROC/Brier were 0.921488/0.756198/0.096856.
The initial selection screen passed, but its speed advantage over 8,192/two
is small and remains a one-pass result. Evidence: `fp8_mlp_b8192_s8/`.

## Completed Blackwell projection-shape attribution

The requested eight-row/sixteen-step BF16 shape-aware capture contained 13,328
GPU events and matched all 2,448 GEMM/GEMV calls, with **zero unmatched GEMM
time**. Summed GPU duration was 0.937302 s, of which 0.665634 s was mapped
matrix multiplication. Shapes were checked against the unchanged model/fusion
mapping, including the shared attention-output shape.

| Projection | Share of all sampled GPU kernel time |
| --- | ---: |
| MLP gate/up | 28.81% |
| MLP down | 14.40% |
| GDN QKV/Z input | 15.18% |
| Shared attention output | 6.81% |
| Full-attention QKV/query gate | 3.95% |
| Vocabulary head | 1.53% |
| GDN B/A input | 0.34% |

The MLP pair accounts for **43.20%**, versus 47.21% in the historical RTX 4080
shape capture. As an illustrative inference from this partial window, speeding
only 43.20% by 3x gives `1 / (0.568 + 0.432/3) = 1.405x`. This explains why
roughly 3x isolated linear-kernel gains can coexist with roughly 1.4x complete
FP4 diagnostic scoring gains. It is not a calibrated full-run performance model
or hardware-counter evidence of compute versus memory limitation.

The historical 4080 **2.767x/3.756x INT4** numbers were complete individual
kernel pipelines with synthetic inputs and reused/preallocated buffers, not a
native W4A4 vLLM inference run. Blackwell's comparable FP4 windows were
2.964x/2.264x on real calibration inputs with online allocations included. The
32-row BF16 serving baseline was already roughly 3.61x faster here (9.947606 s
versus 35.9089 s). Different formats, inputs, allocation policies and uncontrolled
single-window thermals prevent interpreting small cross-GPU ratio differences.

Profile scores had mean/max drift 0.003884/0.029793 against corresponding
baseline rows, correlation 0.999641 and zero flips; profiling/selection changes
remain separate from serving timing and quality selection. Evidence:
`profile_bf16_shapes/shape_summary.json`, its checksummed trace/config/selection,
and the earlier RTX 4080 records in `experiments/local_inference/README.md`.

## Proposed quantization-aware distillation follow-up

The user raised QAT during this campaign. A proposed task-focused pilot would
use 512–2,048 representative **training** trajectories plus separate validation,
not the serving development or confirmation rows. At the existing workload's
average prompt length this is about 6–23 million input tokens per pass; this is
a planning range, not a demonstrated data requirement. Our eight calibration
trajectories establish only a small fitting screen and are insufficient evidence
for end-to-end training.

Freeze the original BF16 monitor as a local teacher, cache its binary decision
distribution with full prompt/model/settings provenance, and train higher
precision master parameters or adapters while fake-quantizing merged weights
and activations to the deployed NVFP4 block/scaling/rounding rules. Matching
the final binary distribution targets monitoring-score preservation without
requiring new human labels. Cache teacher targets separately from hard labels;
keep gradients and FP32 master adapters in higher precision. Export actual
native FP4 tensors and recheck vLLM score agreement after training.

This proposal follows the distinction between task-loss QAT and teacher-KL
quantization-aware distillation in the
[NVFP4 QAD report](https://arxiv.org/abs/2601.20088) and the prepare/convert
description in [PyTorch's QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html).
It is narrower than general language-model capability recovery. Exact training
and serving scale/chunk alignment, straight-through gradient behavior, kernel
availability, memory and throughput need a fresh preflight. No QAT run or new
dependency installation has been launched in this inference campaign.

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
