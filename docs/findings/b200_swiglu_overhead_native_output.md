# SwiGLU overhead and direct FP4 output follow-up

Date: 2026-10-06. The user authorizes two sequential stages: remove overhead
from the existing fused producer, then try native FP4 output. Keep the selected
[attention-FP4 reference](../decisions/b200_attention_fp4_inference_baseline.md)
and its saved controls: 193,224 warm c128 input tokens/s, 156.82 ms warm c1
median and macro/pooled development AUROC 0.878553/0.889729. Use the existing
NC2 B200; retire API/engine 93539/93625 before changing kernels. Keep the pod,
merged weights, master adapters and shared caches. No final-ID selection.

## Stage one: symbolic rows and in-epilogue inverse scaling

Generate an isolated licensed copy of NVIDIA's pinned SwiGLU kernel, retaining
the prior N192 scale-layout correction and inference-only store removal.
One symbolic-M plan accepts all nine native row counts, including M1537.
Native TMA handles partial row tiles without the old allocation/zero/copy
padding sequence. Read the existing activation and frozen-weight inverses
directly inside the epilogue, calculating their reciprocal once per thread's
output row. Preserve raw BF16 GEMM rounding, FP32 descale followed by BF16
rounding, and the activated BF16 boundary before the existing whole-row packer.

Preserve `fp4_swiglu_overhead01`: compilation rejects the unit-dimension stride
of the synthetic FP4 descriptor before GPU arithmetic. Correct the descriptor
to the canonical product stride. `overhead02` passes all nine cases with one
compile but is approximately twice as slow because its source places scaling
division inside the per-value loop. This is a suspected implementation cost,
not a hardware-counter diagnosis. Hoist scale calculation and use the matching
approximate reciprocal in `overhead03`.

`fp4_swiglu_overhead03` passes all nine cases: rows
1/17/129/1536/1537/2304/4096/29184/32768. Zero/extreme rows, isolation and
changed-input CUDA-graph replay pass at the unchanged 1% native ceiling.
Down-projection outputs are exactly equal to the existing producer in all
fixtures. Only one native compile occurs across row counts; no padding is used.
Complete producer speed ratios against FROST gate/up plus fused SiLU/packing:

| Rows | Speed ratio |
| ---: | ---: |
| 1 | 0.8203 |
| 17 | 0.9975 |
| 129 | 0.9994 |
| 1,536 | 1.1132 |
| 1,537 | 1.0230 |
| 2,304 | 1.0891 |
| 4,096 | 1.0617 |
| 29,184 | 1.0561 |
| 32,768 | 1.0413 |

Ratios above one are faster. Five samples of 32 CUDA-graph replays after five
warmups; packing and descale are included, builds and frozen-weight packing
excluded. These are operator gains against the selected producer, not gains
against the earlier fusion trial or serving claims.

`fp4_swiglu_overhead.json` retains the attention-FP4 stack and its tuned plans,
uses this fused producer for M>=1536, and preserves the recorded reference path
for smaller rows. The worker compiles its single native plan before capture;
source-bound admission and live dispatch from all 32 MLPs remain required.
The associated serving screen starts as driver 94653, API/engine 94692/94763.
Its score canary exactly matches the selected reference (mean error zero,
correlation one). All 32 fused and 32 small-row reference routes are observed;
the live audit confirms a single native compile.

## Stage-one serving result

Five fully warm c128 rates are 193,842 / 196,597 / 195,933 / 195,552 / 195,096
input tokens/s. Retain every repeat and the excluded warmup. Controls are the
archived attention-FP4 confirmations on the same retained B200, without an
interleaved control replay.

| Warm measure | Selected reference | Overhead revision | Change |
| --- | ---: | ---: | ---: |
| c128 median input tokens/s | 193,224 | 195,552 | +1.20% |
| c128 aggregate input tokens/s, all five repeats | 193,243 | 195,400 | +1.12% |
| c1 median latency | 156.82 ms | 154.87 ms | -1.24% |
| c1 p95 latency | 287.55 ms | 269.13 ms | -6.41% |
| Source-macro AUROC | 0.878553 | 0.878631 | +0.0078 percentage points |
| Pooled AUROC | 0.889729 | 0.889983 | +0.0254 percentage points |

Repeat-median mean/max score differences are 0.000783/0.040275. One 0.5
threshold flip is on a baseline-unstable `bash_arena` example; two baseline
examples vary around that threshold. Only `bash_arena` source AUROC changes,
by +0.002268. Preserve complete calibration, pAUROC, ties and per-source metrics
in the confirmation summary. These tiny differences do not establish quality
improvement. Strict master precision remains failed separately.

The initial fourteen-pass sweep completes thirteen passes before an HTTP
`ReadError` in its last c128 repeat. The engine stays healthy and idle. Preserve
the failure and partial report. A fresh client completes the missing repeat in
`fp4_swiglu_overhead_sweep_resume02`, binding the original report/failure hashes.
The preceding resume01 saves predictions but its client receipt assertion uses
a nonexistent `prompt_tokens` summary field; preserve that failure, with no
speed claim. Independent fresh-client warm and latency confirmations complete
on the same worker, without repeating startup checks or restarting.

Artifacts and live audits/logs are collected under
`fp4_swiglu_overhead_collection01`. Stop the overhead API/engine before stage
two. The selected default/reference remains unchanged; this is a small measured
gain rather than a statistically established production improvement.

## Stage two: direct packed output

Prepared intervention: emit 16-element E4M3 scales and hardware E2M1 packed
bytes from the native SwiGLU epilogue, removing activated BF16 stores/reads and
the separate pack launch. The first candidate uses a fixed unit global inverse
instead of the reference's whole-row dynamic inverse. This changes quantization
and is explicitly separate from stage one's arithmetic-preserving overhead
work. Bind independent mathematical packing and actual-decoded GEMM references;
retain baseline precision separately from native arithmetic admission. Serving
and development AUROC follow only after supported/finite native validation.

The first native-output source compiles and executes, but canary01's checker
compares a 128-column decoded slice with the full-width mathematical reference.
Preserve the failed harness receipt and generated source. Correct only the
checker to compare all 9,216 columns, then retry as canary02. The separate
decoded-operand GEMM check uses all K columns and the first 16 output columns
with TF32 disabled. Both producer-only and complete producer-plus-down timing
include their scale preparation and packing. No serving quality claim follows
from the first harness failure.

Canary02 completes all nine cases: decoded-operand GEMM arithmetic, finite
outputs and isolation pass, but packing reference errors reach 1.315% at
M32768 and 1.178% on the changed M1 input, exceeding the unchanged 1% limit.
Preserve this failure and its diagnostic timings. Implement correctly rounded
scale division and FMA residual correction of reciprocal-based normalization
before hardware E2M1 conversion. The independent reference uses a device
denominator because [pinned Torch 2.11 scalar division](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
can replace division with multiplication by a rounded reciprocal. Keep the
mathematical ties-to-even contract and finite scale saturation explicit.

Canary03 preserves an assembler rejection of an inline negated FMA operand;
use an explicit `neg.f32` instruction. Canary04 then passes all nine cases,
including full-width packing, decoded GEMM, zero/extreme rows, isolation and
changed-input replay. Packing reference and replay relative-L2 errors are zero
in every fixture. Only one symbolic-row plan is compiled. Strict baseline
precision remains failed because fixed global scaling changes quantization.
The resident native worker is PID 95514 with queue
`fp4_swiglu_native_queue01`; it is stopped and GPU vacancy verified before
serving. Preserve all four receipts, exact sources and the stopped queue state.

At M29184/M32768, corrected direct-output producer speed ratios are
1.1009/1.0999; full producer-plus-down ratios are only 1.0292/1.0200.
Complete MLP timing uses the same standard N256 down-projection plan for both
native fixtures; serving retains its selected shape-dependent down tiles.
Do not infer that removing BF16 traffic translates directly into the producer
gain at whole-MLP or HTTP level. Hardware stall/cache counters are unmeasured.

The integrated `fp4_swiglu_native_output.json` retains the original producer
below M1536, the stage-one whole-row producer at M1536..4096, and direct native
FP4 output above M4096. Medium rows favored stage one in native measurements;
the new format is restricted to large rows. Both native plans are precompiled,
with separately bound receipts and live large-row calls required from every
MLP. Thirty-two focused CPU tests and Ruff pass; the selected reference remains
unchanged.

## Direct-output serving result

The initial fourteen-pass sweep completes without an HTTP failure. Preserve
the initial c128 rates 183,906/155,338 input tokens/s and their post-startup
history separately from fully warm confirmation. Five subsequent warm rates
are 198,009 / 196,022 / 196,866 / 196,502 / 196,896 tokens/s; no repeat is
discarded. Each uses the same 320 identities and 1,310,581 input tokens.

| Warm measure | Selected reference | Overhead | Direct FP4 large rows |
| --- | ---: | ---: | ---: |
| c128 median input tokens/s | 193,224 | 195,552 | 196,866 |
| c128 aggregate input tokens/s, all five repeats | 193,243 | 195,400 | 196,857 |
| c1 median latency | 156.82 ms | 154.87 ms | 159.46 ms |
| c1 p95 latency | 287.55 ms | 269.13 ms | 275.92 ms |
| Source-macro AUROC | 0.878553 | 0.878631 | 0.878105 |
| Pooled AUROC | 0.889729 | 0.889983 | 0.885842 |

Direct output improves warm median throughput **1.88%** against the selected
reference and **0.67%** against the overhead stage. Aggregate gain against the
reference is **1.87%**. Median latency is **1.68% slower** than the reference
and **2.96% slower** than overhead; p95 is 4.04% faster than the reference.
These are archived-control comparisons rather than an interleaved replication.

Source-macro/pooled AUROC changes **-0.0448/-0.3887 percentage points** against
the reference. Repeat-median mean/max score differences are 0.042686/0.420646;
there are ten threshold flips, two on baseline-unstable examples. Preserve these
score shifts despite the small aggregate ranking difference. Source-macro
pAUROC@20/Brier are 0.751991/0.151983 versus 0.748985/0.147939; pooled values
are 0.595925/0.148127 versus 0.599051/0.144583. Pooled recall/FPR remain
0.753165/0.197531 and unique scores rise 84 to 85. Per-source losses include
knowledge-report -9.26 points, soft-trigger -4.71 and insider-trading -1.33;
tiny four/six-row groups have changes of up to 25 points. Preserve the full
per-source and repeat diagnostics rather than interpreting macro cancellation
as precision equivalence. No final-ID evaluation or selection is performed.

The twenty-row canary passes strict master-score admission on this sample:
mean master error 0.019734, correlation 0.996962. Its selected-reference mean
error/correlation are 0.008460/0.999239. This does not override the native
changed-format precision failure or inherited MXFP8/vendor strict failures.
Live usage covers all three row routes on all 32 MLPs, with one direct-output
and one overhead plan compiled before capture. Existing attention/preparation/
tuning audits pass on the current engine. The runner also archives stale
`native_swiglu_overhead.json` and legacy `native_swiglu.json` from retired
workers; those are unused by this condition and are not fresh runtime passes.

Both requested stages are implemented and measured as named recipes. Keep the
selected attention-FP4 reference unchanged. Retain the experimental native
output server as API/engine **95619/95657**, port **8010**, healthy and sole GPU
ownership, approximately **170,328 MiB**. Preserve the existing pod and caches;
the separate native worker is stopped. Kernel/source, initial sweep, warm
throughput/latency, ranking/calibration/threshold diagnostics, failures and
runtime logs are collected. Commits 7bfcf59 and 5cb6d02 contain the two features.
