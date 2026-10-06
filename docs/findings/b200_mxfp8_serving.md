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

Pins: vLLM 0.24.0, Torch 2.11.0+cu130, cuDNN frontend 1.31.0/backend 9.26.0.51,
NVIDIA source revision `51d9d06b574222378a3d806009accab098e73705`, Triton 3.7.1
and CuTe DSL 4.8.0. See the [experiment contract](../../experiments/b200_attention_gdn_serving/README.md#forward-only-cudnn-mxfp8-serving).
