# Forward-only cuDNN MXFP8 attention on B200

Date: 2026-10-06. The corrected causal D256/GQA serving bridge completes the
frozen inference screen. Five additional resident concurrency-128 passes give
median **173,938 input tokens/s**, **+4.83%** against the selected FP4-MLP/GDN
baseline's five-pass confirmation of 165,927. After reviewing these results,
the user selects this recipe as the next optimization reference; see the
[selection decision](../decisions/b200_mxfp8_inference_baseline.md). Preserve
the original strict failures and the earlier diagnostic status.

## Scope and comparison

Retain all 64 FROST FP4 MLP projections and 48 large FP4 GDN projections.
Replace only full-attention prefill with NVIDIA cuDNN's SM100 D256 MXFP8
forward kernel, using rowwise Q/K and columnwise V E4M3 payloads with E8M0
scales per 32 elements. A forward-only producer gathers directly from the
BF16 HND paged cache and emits forward payloads and scale atoms. Include
gathering, quantization, allocation and token-offset conversion in timing.
Decode, persistent KV cache and GDN recurrence remain BF16; gates/state remain
FP32. Use the existing ephemeral merged BF16 checkpoint and preserve its FP32
master adapter.

This is an adapted kernel: two mixed polynomial exp2 helpers are replaced with
hardware vector exp2 in a checksum-named derived source. Installed NVIDIA
sources are unchanged. Preserve the native unit P scale and four-log2-unit
running-max update policy. This is not a stock-kernel performance claim.

Reuse one NC2 B200, 90% memory, 128 sequence slots, 32,768 scheduled/context
tokens, disabled prefix caching and one constrained decision token. Compare
saved controls; no control server runs concurrently. Keep all shared compiler
caches. The 320 training-seen systems-development rows contain 1,310,581 input
tokens. These results do not measure final ID generalization.

## Numerical validation and integration correction

Canaries01–03 retain the strict precision failures and the investigation of the
quantized reference. Canary04's derived exp2 kernel matches its independent
quantized reference, but the initial fixture misses the real caller's metadata
contract. Pinned vLLM's `cum_seq_lens_kv` contains cumulative **cache pages**;
cuDNN packed THD expects cumulative **tokens**. The first serving bridge used
the former as the latter. Preserve `fp4_gdn_cudnn_mxfp8_01` and
`mxfp8_confirmation01` with separate `invalidation.json` records. Their speed
and quality numbers are invalid evidence about MXFP8.

Retire that server before correcting the bridge. Derive token offsets on-device
from exact `seq_lens`, including partial final pages. Canary05 supplies actual
page-count metadata, asymmetric token lengths and shuffled HND pages, tests
batch 128/context 32,768, then changes lengths and pages during graph replay.
All seven producer/arithmetic/replay cases pass, with maximum quantized-reference
relative-L2 **0.001667**. The unchanged strict FP32-reference ceiling of 5%
still fails: maximum **5.336%**. `passed=false`, `arithmetic_passed=true` and
`diagnostic_only=true` remain distinct. Earlier receipts cannot authorize the
corrected worker. Runtime batch count also avoids compilation per batch count.

The corrected model screen passes all 112 FP4 projection checks. Its twenty-row
score canary passes baseline-relative acceptance: mean difference **0.012958**,
correlation **0.998448**. Strict master-score parity remains failed. The HTTP
summary's diagnostic flag describes score acceptance only; it does not erase
the native strict-precision diagnostic classification.

## Throughput and development quality

The complete screen includes six quick passes and eight full-cohort passes.
Full-sweep median input token rates at c16/32/64/128 are
**135,324 / 138,625 / 178,719 / 165,440**. The c32 repeats vary substantially;
do not treat their median as a settled warmed result.

One further full c128 warmup is excluded from the five-pass confirmation.
The five timed rates are **155,348 / 158,069 / 173,938 / 175,427 / 176,857**
tokens/s, so the +4.83% median gain is modest and retains timing variation.
A client `httpx.ReadError` occurs after two complete passes; preserve that log
and resume the three missing passes with fresh clients on the same healthy
API/engine PIDs. Do not reload the model or discard the first two passes.

Using repeat-median scores, source-macro AUROC is **0.884634 → 0.901413**
(**+1.68 percentage points**); pooled AUROC is **0.904087 → 0.900356**
(**−0.37 points**). Source-macro partial AUROC at 20% FPR is
0.780666 → 0.794379. Pooled Brier worsens 0.134088 → 0.137526 and pooled
FPR at threshold 0.5 rises 0.160494 → 0.191358. Full-cohort mean/max score
drift is **0.046336 / 0.456479**, with **20 threshold flips**. Keep all per-source
results and score-tie diagnostics; small source groups make macro movements
coarse, and improved macro AUROC does not establish better generalization.

## Artifacts and reuse

Results: `results/b200_attention_gdn_serving/fp4_gdn_cudnn_mxfp8_02`,
`cudnn_mxfp8_canary05.json`, `mxfp8_confirmation02`, and the checksum-bound
`mxfp8_serving_collection02`. Logs remain under the matching Runpod log tree.
Executed source archives and all matched IDs/prompt hashes/token counts are
retained. Twelve focused tests, Ruff, native cases and finite-score checks pass.

The experimental server is retained warm: API **84940**, engine **85056**, port
8010; configuration and shared cache paths are recorded in `server.json` and
`campaign.json`. Stop it before modifying active kernels/settings. The B200 pod
remains running. No new billable capacity is launched. The user's subsequent
reference selection is recorded separately in `quality_acceptance.json`.

## Profile of the selected reference

`mxfp8_profile01` runs a separate full 320-row c128 pass on the same resident
worker, with no kernel/settings changes or control replay. Its 7.602-second
request duration includes profiler overhead and is excluded from speed claims.
The trace records 46,704 kernels, 6.696 seconds of summed kernel time, a
6.691-second interval union and a 7.279-second first-to-last-kernel window.
GPU occupancy within that window is **91.92%**. Gaps are not automatically
CPU dispatch time; HTTP/tokenization and overlapping host work are separate.

| Exclusive CUDA category | Seconds summed | Share of kernel time |
| --- | ---: | ---: |
| FP4 GEMMs | 1.527 | 22.81% |
| FP4 activation packing and scales | 0.804 | 12.01% |
| Fused elementwise, normalization, gates and layouts | 1.785 | 26.66% |
| GDN core | 0.857 | 12.80% |
| MXFP8 attention core | 0.620 | 9.26% |
| MXFP8 gathering, quantization and offsets | 0.080 | 1.20% |
| Remaining BF16 GEMMs | 0.504 | 7.53% |
| Causal convolution | 0.395 | 5.90% |
| Other kernels | 0.123 | 1.84% |

FP4 GEMMs and preparation together account for **34.82%**. Runtime preparation
is activation work; frozen weights are already packed. `_row_inverse` costs
0.255 seconds, `_pack_row_blocks` 0.538 and `_row_scale` 0.012, each with
4,704 calls (112 projections × 42 model iterations). The GEMM epilogue already
fuses output descaling. The next suggested screen is row-amax/activation-packing
fusion and fusion with preceding activation/normalization producers, preserving
row-local scales and BF16 rounding. The broad 26.66% fused category contains
multiple operations and is not one removable kernel. Its largest individual
kernel is the fused MLP activation chain, approximately 0.307 seconds.

Full attention including preparation accounts for **10.46%**, so it is no
longer the dominant block. MXFP8 preparation alone is only 1.20%; prioritize
the larger FP4 preparation cost first. GDN core and its surrounding fused
normalization/gating work remain another substantive target. Native GDN backend
failures remain preserved; the profile does not validate a replacement.

Raw trace, profiler table, predictions, kernel counts/durations, exclusive
category membership, CPU operator totals and GPU interval calculation are
retained under `results/b200_attention_gdn_serving/mxfp8_profile01`. CPU totals
are nested and overlap GPU work, so they are not wall-time fractions. This is
one throughput profile, not a latency-at-c1 profile or an optimization result.

## FP4 preparation diagnosis and vendor-kernel candidates

The subsequent investigation leaves API 84940 / engine 85056 unchanged. Reconcile
all 4,704 row-reduction, packing and output-scale launches from the saved trace.
The row-reduction grid gives M; the packing grid gives
`ceil(M/128) * K/16`. Pair launches in stream order and verify the reconstructed
widths against the actual model: 2,560 for MLP gate/up and GDN inputs, 4,096 for
GDN output, and 9,216 for MLP down. Shape totals and all three timing totals
exactly reconcile with the raw kernel summary.

| Activation width | Calls | Preparation time | Share of preparation |
| --- | ---: | ---: | ---: |
| 2,560 | 2,352 | 245.49 ms | 30.53% |
| 4,096 | 1,008 | 146.57 ms | 18.23% |
| 9,216 | 1,344 | 411.94 ms | 51.24% |

The current path materializes BF16 producer outputs, scans each entire row for
its dynamic maximum, then reads the activation again to compute 16-element
block scales and hardware FP4 codes. The GEMM is a custom operator, so compiler
fusion of preceding normalization/SwiGLU does not reach into its internal
packing. This is a producer/quantizer boundary cost, not repeated weight packing
or a missing hardware FP4 conversion. Output descaling already runs inside the
FROST GEMM epilogue. Its separate `_row_scale` helper costs only 11.66 ms,
0.17% of kernel time; `_row_inverse` and `_pack_row_blocks` cost 254.81 and
537.54 ms. Prioritize those larger stages.

The reconstructed input scans total **2.813 TB of logical reads** over the full
320-row workload. This is not measured HBM traffic: caches may serve reads, and
the estimate omits additional implementation traffic. Logical scan throughput
at K9,216 is 6.67 TB/s; packing input-plus-code/scale-output throughput is
3.38 TB/s. These observations support investigating memory movement plus
reduction/conversion work, but do not establish a roofline bottleneck without
memory/instruction counters. Current packing uses 37–39 registers/thread in
trace metadata; reduction uses 16–28. Do not interpret the profiler's estimated
occupancy field as a measured occupancy counter.

The [earlier native FP4 study](b200_mlp_gemm.md#chunked-per-row-packing-complete-native-screen)
already compared fused-row and chunked packing. The old fused-row path uses
155 registers at K9,216, versus roughly 38 for chunked packing, with no spills
reported in that study. A single whole-row fusion is therefore not automatically
an improvement. Preserve low-register tiling or adopt a tuned vectorized vendor
implementation; do not simply repeat the old whole-row prototype.

NVIDIA documents the gap between low-precision GEMM-only performance and complete
performance including quantization, including the extra maximum-reduction pass.
Its delayed-scaling example is FP8 and is not evidence that our row-scaled NVFP4
contract supports delayed scaling unchanged. [NVIDIA performance study](https://developer.nvidia.com/blog/how-to-optimize-transformer-based-models-for-low-precision-training/).

The installed FlashInfer **0.6.12** already exposes
`nvfp4_quantize(..., backend="cuda", per_token_activation=True)`, with E4M3
block-16 scales, selectable 128×4 layout and returned FP32 token scales. This
is the first suggested bounded comparison against the current packer while
keeping FROST GEMMs fixed. Use the documented inverse base multiplier as a
host float: the pinned wrapper calls `.item()` if it receives a tensor, which
would synchronize a GPU scalar and break graph capture. Its per-token CuTe
backend is rejected by the pinned wrapper; do not assume newer documentation's
CuTe symbol exists locally. API/format compatibility does not establish
bit-exact codes, rounding, tail initialization, graph replay or a speed gain.
[Pinned source](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.12/flashinfer/quantization/fp4_quantization.py),
[current API](https://docs.flashinfer.ai/generated/flashinfer.quantization.nvfp4_quantize.html).

For subsequent producer fusion, FlashInfer has fused RMSNorm→NVFP4 and
SiLU/multiply→NVFP4 implementations. Pinned RMSNorm FP4 and the expert activation
fusion are present; the newer dense `silu_and_mul_nvfp4_quantize` symbol is absent
from the pinned quantization module. Their supplied scalar-scale contracts differ
from our freshly calculated row scales. Adapting the producer to preserve row
maxima, BF16 boundaries and the FROST scale layout is a separate intervention;
calibrated fixed scales would change quantization policy and need a quality screen.
[RMSNorm API](https://docs.flashinfer.ai/generated/flashinfer.cute_dsl.rmsnorm_fp4quant.html),
[dense SwiGLU API](https://docs.flashinfer.ai/generated/flashinfer.quantization.silu_and_mul_nvfp4_quantize.html).

Recommended order: compare the existing vendor CUDA per-token quantizer at the
three reconstructed widths; then investigate a row-scaled SwiGLU/packing
producer for the dominant K9,216 down-projection path; then normalization/packing
producers. Preserve zero-row/extreme-value handling, actual scales and rounding,
128×4 tails, supported shapes and changed-input graph replay. Any serving kernel
change requires stopping the old server, matched token throughput and AUROC.
No replacement is integrated or benchmarked by this source/trace investigation.
Halving the measured preparation cost alone would suggest roughly 5–6% complete
throughput improvement under additive-cost assumptions, not a demonstrated gain.

Artifacts: `mxfp8_profile01/diagnose_fp4_preparation.py` and
`fp4_preparation_diagnosis.json`, bound to the original trace checksum. The
baseline selection and resident worker remain intact.

Pins: vLLM 0.24.0, Torch 2.11.0+cu130, cuDNN frontend 1.31.0/backend 9.26.0.51,
NVIDIA source revision `51d9d06b574222378a3d806009accab098e73705`, Triton 3.7.1
and CuTe DSL 4.8.0. See the [experiment contract](../../experiments/b200_attention_gdn_serving/README.md#forward-only-cudnn-mxfp8-serving).
