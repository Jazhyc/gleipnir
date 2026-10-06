# Native FP4 gate/up GEMM and SwiGLU fusion

Date: 2026-10-06. User-authorized follow-up to the matched GEMM backend
comparison. **Keep the default unchanged:** fusion improves warm median
throughput slightly, but aggregate throughput and interactive latency worsen.
Keep it as a named experimental recipe. Use the existing NC2 B200 and stop its prior serving process
before native work. Preserve master adapters, merged BF16 source weights,
shared caches, historical failures and archived control predictions.

Adapt NVIDIA's pinned SM100 block-scaled dense SwiGLU template, retaining
its license in the separately generated source artifact. Interleave existing
FP4 weight payloads and block-scale bytes into 32-wide up/gate pairs without
requantization. The permutation preserves each 16-row frozen scale group.
Round raw GEMM values to BF16, apply the same per-row FP32 descale, round to
BF16 again, then compute SwiGLU and emit its half-width BF16 activation.
The existing rowwise emitter packs that activation with whole-row amax,
E4M3 block scales and hardware FP4 RNE. This is not direct FP4 GEMM output.
Remove the full-width training-only gate/up write, backward shared storage
and its store pipeline. An unused tensor descriptor retains the vendor API's
dtype/layout planning with a single row of storage.

## Native results and failed attempts

The corrected, shared serving packer is validated in `fp4_swiglu_compare07`.
Both selected N192 variants pass all eight row counts, including zero rows,
an extreme second row, isolation and changed-input CUDA-graph replay. Frozen
weight permutation matches exactly. Every down-projection relative-L2 error
is zero, at the unchanged 1% admission ceiling. Tiny activation differences
from exponential/reciprocal arithmetic remain recorded separately.

Selected MMA tile 256×192, cluster 2×1 and packed FP32 epilogue arithmetic:

| Rows | Reference gate/up + SiLU/packing, ms | Fused producer + packing, ms | Speed ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.019776 | 0.024681 | 0.8013 |
| 17 | 0.018583 | 0.024724 | 0.7516 |
| 129 | 0.018626 | 0.024883 | 0.7485 |
| 1,536 | 0.059670 | 0.055642 | 1.0724 |
| 2,304 | 0.084066 | 0.078172 | 1.0754 |
| 4,096 | 0.137113 | 0.131228 | 1.0448 |
| 29,184 | 0.986951 | 0.939208 | 1.0508 |
| 32,768 | 1.111725 | 1.062189 | 1.0466 |

Ratios greater than one are faster. Medians of five samples of 32 CUDA-graph
replays after five graph warmups; both producers include descaling and packing.
Builds, weight permutation and input packing are excluded. These are native
operator measurements, not HTTP serving gains. Preserve the sizable sample
variation in the large-row receipts; use warmed full-model confirmation.

Preserve seven exact source archives and all 148 executed check records:
120 pass and 28 fail. Initial four cases fail on M/K/batch descriptor order;
the next four fail because the adapter omitted optional FFI placeholders.
The first executable attempt passes eight large-row checks, then a one-row
launch raises a CUDA illegal instruction and three later cases inherit that
failed context. Pad rows to the native tile size and include all padding/copy
work in timing. The 32-case scalar follow-up passes, but large-row gains are
only 0–1.5%. Packed arithmetic and larger clusters pass 48 of 64 cases; the
two N192 variants fail 16 numerical checks with approximately 54–58%
down-projection error. Never accept those timings as a valid gain.

The [NVIDIA template](https://github.com/NVIDIA/cudnn-frontend/blob/main/python/cudnn/gemm/cutedsl/dense/swiglu/dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py)
tests the N192 scale-layout correction against half-width C rather than the
input MMA width. For advertised input N192, C has width 96, so that branch is
unreachable. In an isolated generated copy, key it on the input tile width.
The targeted follow-up passes all 16 cases; the final shared-packer follow-up
also passes all 16. Record this source correction, both upstream/generated
hashes and all prior failed receipts. Installed vendor kernels are unchanged.

## Serving intervention

`fp4_swiglu_fused.json` uses the corrected 256×192/2×1 vector kernel for
physical rows ≥1,536. Smaller batches explicitly retain the selected FROST
gate/up plus fused SiLU/packing producer, avoiding the measured padding loss.
The other three projection shapes retain their selected row-dependent FROST
tiles; combined normalization/vendor preparation and cuDNN MXFP8 attention
are unchanged. A custom operator keeps row dispatch inside the opaque native
boundary, preventing a model graph branch on the threshold. Fail if a new
native plan is encountered during graph capture.

Require live fused calls from every one of the 32 MLPs, the current worker PID,
the exact generated kernel hash and the bound native receipt. The existing
preparation coverage remains required, with the fused stage recorded as SiLU
production. Startup imports, full-model compilation and adapter score canary
are separate from warm serving measurements. Reuse the archived selected
reference: 182,681.956 input tokens/s at c128 and 146.9997 ms warm c1 p50.
Compare warm full-cohort scores with its five confirmation repeats; also
retain the initial full sweep and its two-repeat comparisons. Frozen 320 rows
are training-seen systems development data; no final ID selection is performed.

## Warm serving result

The five c128 confirmation rates are 184,804 / 186,096 / 185,963 / 154,182 /
186,626 input tokens/s. Preserve the slower fourth repeat, including its
8.5002-second duration and latency excursion; its cause is not independently
diagnosed. The existing reference's five rates are 180,740 / 183,330 /
182,682 / 182,902 / 181,867. These are archived-control comparisons on the
same retained B200, not a new interleaved control replay.

| Warm measure | Selected reference | Fused producer | Change |
| --- | ---: | ---: | ---: |
| c128 median input tokens/s | 182,682 | 185,963 | +1.80% |
| c128 total tokens / total time, all five passes | 182,300 | 178,531 | −2.07% |
| c1 p50 latency, median of two passes | 147.00 ms | 155.84 ms | +6.02%, slower |
| c1 p95 latency, median of two passes | 270.26 ms | 287.02 ms | +6.20%, slower |
| Source-macro AUROC | 0.884608 | 0.884647 | +0.0039 percentage points |
| Pooled AUROC | 0.908150 | 0.908208 | +0.0059 percentage points |

All five 320-row confirmation passes use 1,310,581 input tokens each. The
aggregate is 6,552,905 tokens / summed measured duration, not the arithmetic
mean of per-pass rates. Threshold flips between repeat-median scores are zero;
mean/max score differences are 0.000553/0.029793. Twenty-nine dual-label sources
have defined AUROC, with one undefined group. Only `bash_arena` changes source
AUROC, by +0.001134. Source-macro pAUROC@20 changes 0.778416→0.778612 and Brier
0.139499→0.139491; pooled pAUROC@20 changes 0.644971→0.645281 and Brier
0.130132→0.130104. Unique pooled scores change 87→88. These tiny differences
are not evidence of a monitoring-quality improvement.

The twenty-row scoring canary matches the selected reference exactly: mean
error zero, correlation one, with nonzero adapter effect 0.827394. Strict
master parity remains failed and explicitly finite/baseline-relative acceptance
is preserved. Live preparation, attention and fused-MLP audits pass for engine
92580; fused dispatch covers all 32 MLPs and the small-row reference route is
also observed. The initial load sweep completes seven of eight full passes
before an HTTP `ReadError` in its final repeat. Preserve its failure receipt and
partial summary. The engine stays healthy, idle and responsive with no logged
GPU failure. Fresh client sessions complete five full confirmation passes and
two further warm c1 passes without restarting the server or rerunning canaries.
The runner also copies the old `native_gemm_tuning.json` from retired engine
91317; that inherited receipt is unused by this condition and is not a fresh
runtime pass. The new worker binds the selected reference tile receipt and
records fused dispatch separately in `native_swiglu.json`.

Startup takes about eleven minutes, including a new model graph compile.
Repeated Triton argument-analysis warnings in the existing QK/RoPE kernel are
nonfatal; they also occur in the original reference's compile log. Early
serving passes have substantial first-touch latency spikes. The adapted native
plans specialize on padded M and reuse resident/disk caches afterward. The
native arithmetic receipt alone therefore does not imply good production
startup behavior or tail latency. A modest median gain, an unexplained warm
excursion and a latency regression do not justify replacing the default.

Retain the named fused worker as API **92464**, engine **92580**, port **8010**,
healthy with sole GPU ownership, approximately 168,974 MiB and 33°C at collection.
The pod, shared caches and selected reference remain unchanged. The campaign
receipt records the active experimental configuration and live PIDs. Artifacts
include seven native receipts/source archives, the partial failed load sweep,
warm confirmation/latency predictions, runtime audits, executed code, logs and
`fp4_swiglu_analysis01.json`. All 199 collection hashes verify locally.
Forty-one focused CPU tests and Ruff pass; scoped implementation commits retain
the experimental recipe rather than silently promoting it.
