# FP4 activation preparation on the MXFP8 serving reference

Date: 2026-10-06. The user authorizes all three proposed preparation changes:
installed CUDA per-token NVFP4 packing, fused SwiGLU packing, and fused
post-attention residual/RMSNorm packing. All three independent trials and their
combination complete. The combined stack gives a five-pass median of
**185,767 input tokens/s**, **+6.80%** over the accepted MXFP8 reference's
173,938, while interactive median latency stays at 149.5 ms. It passes the
baseline-relative score canary; native/master strict failures remain separate.
Source-macro/pooled AUROC changes are **−1.61/+0.81 percentage points**.
Keep the combined worker warm and leave the selected reference unchanged.

| Preparation change | Warm input tokens/s | Speed change | c1 median ms | Source-macro AUROC change, pp | Pooled AUROC change, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected reference | 173,938 | — | 149.50 | — | — |
| Vendor packing | 181,786 | +4.51% | 150.77 | −1.00 | +0.47 |
| SwiGLU fusion | 180,903 | +4.00% | 147.12 | −1.83 | −0.39 |
| Normalization fusion | 178,260 | +2.48% | 147.44 | −1.08 | +0.05 |
| Combined | 185,767 | +6.80% | 149.47 | −1.61 | +0.81 |

## Frozen scope and evidence

Use the existing NC2 B200 pod, ephemeral merged BF16 model, all 64 FP4 MLP and
48 large FP4 GDN projections, adapted cuDNN MXFP8 full-attention prefill,
BF16 cache/decode/recurrence and FP32 gates/state. Preserve the FP32 master.
Keep 90% GPU memory, 128 sequence slots, 32,768 scheduled/context tokens,
disabled prefix caching and the constrained one-token scoring contract.
Retire each server before changing kernels. No concurrent control replay or
new billable capacity is required.

The frozen 320 training-seen systems-development rows contain 1,310,581 input
tokens; manifest SHA256 is
`b02af232d76935f2cda2a52213ebce27ae04ca5aaf44898cfe496104adad4266`.
Quality compares saved c128 repeat-median scores from
`fp4_gdn_cudnn_mxfp8_02`. Speed compares the five-pass median in
`mxfp8_confirmation02`. These score and speed references are separate artifacts;
batch-dependent score ties mean quality can differ slightly from that earlier
five-pass quality summary. No result measures final ID generalization.

## Native admission

`fp4_prepare_vendor05` tests K2560/4096/9216 and M1/17/129/32768, including zero
and extreme rows, row isolation and changed-input graph replay. Installed
FlashInfer 0.6.12 emits invalid scales for all-zero rows, so explicitly initialize
their mathematical zero payload and scales, as well as scale padding. Convert
per-token scales to the inverse convention required by FROST. A host scalar
avoids the wrapper's GPU `.item()` path. Do not fall back to another quantizer.

All twelve actual-operand GEMM arithmetic/finite checks pass, with maximum
relative-L2 1.08e-5. The unchanged 1% baseline decoded-precision ceiling still
fails at 1.81%; `passed=false` and `arithmetic_passed=true` remain distinct.
Long-row packing times are 0.0926→0.0462, 0.1354→0.0619 and 0.2892→0.1294 ms
at the three widths. Preserve the earlier missing-PATH, padding-dtype,
zero-row-nonfinite and strict precision failures.

`fp4_fusion_canary01_silu` preserves the failed eager-style BF16 intermediate
rounding attempt. Generated Inductor code instead retains FP32 SiLU and
multiplication until a single BF16 output. Canary02 follows that compiled
contract. Both fusions pass eight cases, native actual-operand GEMM checks,
isolation and changed-input graph replay. SwiGLU decoded values/payloads/scales
are exact in the tested cases. Normalization preserves the BF16 residual exactly;
maximum decoded relative-L2 is 0.000677 and native GEMM error 0.0000923.

At M32768, select eight warps for SwiGLU: producer plus packing falls from
0.806 to 0.457 ms, 1.76 times faster. Select four warps for normalization:
0.499 to 0.110 ms, 4.55 times faster. Native timings use CUDA graphs and include
allocations/layout/inverse work, but exclude HTTP and host dispatch. Each
fusion bypasses the existing opaque pack-and-GEMM boundary with explicit
packed tensors and registered custom operations/fake implementations.

## Vendor end-to-end result

`fp4_prepare_vendor_serving01` completes six quick latency passes and eight
full-cohort throughput passes. Five additional warmed c128 passes in
`fp4_prepare_vendor_confirmation01` reuse the same API/engine PIDs 86965/87130.
Their input rates are 180,718 / 182,389 / 180,956 / 182,640 / 181,786 tokens/s;
exclude the preceding full-cohort warmup. The c1 quick median latency is
150.77 ms versus 149.50 ms for the saved reference, so this change does not
improve interactive median latency. Keep sweep variability separately.

The twenty-row score canary's baseline mean difference is 0.021804 and
correlation 0.993867; baseline-relative and strict master acceptance both fail.
Finite diagnostic continuation does not erase either failure. Full-cohort
source-macro AUROC is 0.900666→0.890636, **−1.00 percentage point**; pooled
AUROC is 0.900238→0.904927, **+0.47 points**. Mean/max score differences are
0.046344/0.442974, with twenty threshold flips. Pooled Brier is
0.137661→0.133417 and FPR at 0.5 is 0.179012→0.160494. Preserve per-source,
partial-AUROC and score-tie diagnostics; small source groups make macro changes
coarse. The worker is retired before the independent SwiGLU trial starts.

## Independent SwiGLU model result

`fp4_prepare_silu_serving01` completes all fourteen passes and passes the
baseline-relative score canary: mean difference 0.014455, correlation 0.998019.
Strict master parity remains failed. Five warm passes in
`fp4_prepare_silu_confirmation01` give median **180,903 input tokens/s**,
**+4.00%**, with rates 179,499 / 181,522 / 181,014 / 180,903 / 179,092.
API/engine PIDs 87814/87891 are reused and subsequently retired.

Source-macro AUROC is 0.900666→0.882396, **−1.83 percentage points**; pooled
AUROC is 0.900238→0.896331, **−0.39 points**. Mean/max score differences are
0.037631/0.330534, with thirteen threshold flips. Native bitwise agreement in
random/extreme fixtures does not establish whole-model score identity.
Inspecting cached serving kernels confirms FP32 `gate / (1 + exp(-gate))`
and multiplication followed by BF16 storage, matching the canary's formula;
the cause of the model-level drift is not established. Do not promote a claim
of exact model equivalence from these fixtures.

## Independent normalization model result

`fp4_prepare_norm_serving01` completes all fourteen passes, passes the new
current-worker/count audit and passes the baseline-relative score canary:
mean difference 0.015047, correlation 0.998144. Strict master parity remains
failed. Five warm passes in `fp4_prepare_norm_confirmation01` give median
**178,260 input tokens/s**, **+2.48%**, with rates
178,073 / 177,898 / 179,341 / 178,260 / 178,721. Interactive c1 median is
147.44 ms versus 149.50 ms. API/engine PIDs 88396/88456 are reused, then retired.

Source-macro AUROC is 0.900666→0.889885, **−1.08 percentage points**; pooled
AUROC is 0.900238→0.900727, **+0.05 points**. Mean/max score differences are
0.045887/0.411570, with eighteen threshold flips. The large isolated native
producer gain becomes a modest full-model gain; do not extrapolate kernel
speed ratios to the complete serving stack. All three independent trials
support a bounded combined diagnostic screen, without promoting their drift.

## Combined model result

`fp4_prepare_combined_serving01` completes all fourteen passes. The current
worker's native preparation receipt verifies 32 normalization, 32 SwiGLU and
48 vendor-packing calls, with all three validation hashes bound. Strict native
preparation precision remains failed because the vendor receipt's 1.81% result
is not erased. The baseline-relative score canary passes: mean difference
0.012902, correlation 0.998811. Strict master parity remains failed.

Five warm passes in `fp4_prepare_combined_confirmation01` give
185,767 / 186,024 / 186,004 / 184,014 / 185,701 input tokens/s. Exclude the
preceding full-cohort warmup. API/engine PIDs **88949/89008**, port **8010**,
remain resident after the experiment. The measured +6.80% is smaller than the
sum of the individual gains; native kernel ratios do not predict additive
full-model speedups. c1 median/p95 are 149.47/286.38 ms versus the saved
reference's 149.50/299.28 ms; two quick repeats do not establish a latency
improvement. Keep sweep variability and warmed confirmation separate.

Source-macro AUROC is 0.900666→0.884569, **−1.61 percentage points**; pooled
AUROC is 0.900238→0.908326, **+0.81 points**. Mean/max score differences are
0.042684/0.485976, with twenty-two threshold flips. Preserve calibration,
partial-AUROC, per-source and score-tie outputs. Passing a twenty-row canary
does not remove the measured source-average decline or establish equivalence.
Do not replace the user-selected reference based on the pooled gain alone.

## Profile of the combined stack

`fp4_prepare_combined_profile01` profiles a separate full c128 pass on the
same worker. Exclude its 7.156-second request time from speed claims. It records
44,688 kernels, 6.326 seconds summed kernel time, 6.320 seconds of interval
union and a 6.843-second first-to-last-kernel window: 92.36% busy within that
window. The saved reference has 46,704 kernels and 6.696 seconds summed time,
so launches fall 4.32% and summed time falls 5.54%. This is one profile per
stack, not a separately repeated performance benchmark or SM occupancy measure.

| Exclusive combined CUDA category | Seconds | Share |
| --- | ---: | ---: |
| FP4 GEMMs | 1.520 | 24.03% |
| Remaining fused elementwise/norm/gates/layouts | 1.242 | 19.64% |
| GDN core | 0.909 | 14.37% |
| Fused SwiGLU and normalization FP4 producers | 0.822 | 13.00% |
| MXFP8 attention core | 0.611 | 9.65% |
| BF16 GEMMs | 0.504 | 7.97% |
| Causal convolution | 0.391 | 6.18% |
| Other kernels | 0.129 | 2.03% |
| Standalone FP4 packing/scales | 0.119 | 1.88% |
| MXFP8 gather/quantization/offsets | 0.079 | 1.25% |

Standalone FP4 packing/scales fall from 0.804 to 0.119 seconds, but account
for the new fused producers separately. The broader packing plus fused
elementwise/producer group falls from 2.589 to 2.183 seconds, about 15.7%.
SwiGLU emission alone still takes 0.658 seconds, 10.4% of the new kernel sum;
normalization emission takes 0.164 seconds. FP4 GEMMs remain almost unchanged
at 1.520 versus 1.527 seconds. These measured totals explain why reducing
preparation does not multiply whole-model throughput. CPU operator totals
overlap/nest and are not wall-time fractions; retain the raw trace and exclusive
category membership rather than calling `aten::copy_` time a dispatch fraction.

## Cache and artifact handling

The previous server's FlashInfer cache lived under `/root/.cache/flashinfer`.
Copy its 28 MB into the network-volume `.cache/flashinfer` directory and set
`FLASHINFER_WORKSPACE_BASE` to the repository root. Native builds refresh after
path relocation and then share this persistent namespace. Preserve existing
Inductor, vLLM, Triton, CuTe and cuDNN cache locations. Startup imports, builds,
compilation and graph capture are excluded from warmed speed measurements.

Native receipts and executed source archives are collected locally and their
source hashes verified. Serving predictions, score failures, source archives
and warm confirmation receipts are also collected. Outputs remain under
`results/b200_attention_gdn_serving/`, logs under the matching Runpod tree.
Focused tests and Ruff pass. The selected optimization baseline is unchanged.

`fp4_preparation_collection01` binds all result/source artifacts, exact warm
client/profile scripts and frozen log/server/baseline snapshots by checksum.
All 76 timed passes preserve IDs, prompt hashes and token counts with finite
scores; executed source bindings are verified against each archived trial.
Twenty-two focused CPU tests pass. The final HTTP health check returns 200,
only engine PID 89008 owns GPU compute, and the shared cache paths plus actual
Torch/vLLM/FlashInfer/Transformers versions are recorded in `health.json`.
`campaign.json` marks the campaign complete and the combined server retained;
retire it before changing kernels. The B200 pod remains running. No recurring
follow-up is promised after this active turn.
