# FP8 GDN projections on the FP8-attention default

2026-10-08. Replacing only the 48 large GDN projections with FP8 gives a small
ID source-macro pAUROC@20 gain of **0.261 percentage points**, almost entirely
on STRIDE. Gloom is nearly unchanged, while batch input throughput falls about
10%. Calibration improves substantially, but development low-FPR ranking
regresses. This does not support FP4 GDN projections as the main explanation
for the remaining Gloom gap within this recipe. FP8 remains quantized and
interactions are unmeasured; the result does not rule out all GDN contributions.
Keep the selected FP8-attention/FP4-GDN default unchanged.

## Frozen intervention

The [experiment contract](../../experiments/b200_gdn_precision/README.md)
predeclares one candidate, three quick64/c1 and six full320/c128 development
repeats, and one full ID pass regardless of development results. Only the
24 GDN blocks' QKV/Z and output projections change from FP4 to E4M3 FP8 W8A8,
with per-channel weight scales, dynamic per-token activation scales, native
CUTLASS GEMM and BF16 outputs. Quantize original merged BF16 weights directly.
Small gate projections, convolution/recurrence operands and FP32 gates/state
retain their original precision. All 16 attention projections remain FP8,
all 64 MLP projections FP4, with native SwiGLU output and MXFP8 attention core.
Keep the same final 272-update FP4/FA4-trained adapter, merged artifact,
vLLM 0.31 runtime, native frontend/direct FROST, synchronous stock FCFS,
CUDA GDN/automatic context parallelism, prefix caching off and 32K/128 limits.

Reuse the checksum-bound `b200_attention_precision/precision01/fp8` development
and ID results on the same B200 UUID. No unchanged control rerun, training,
threshold fitting, checkpoint search, OOD evaluation or automatic promotion.
The archived full-BF16 dynamic-LoRA result remains a whole-stack comparison,
with host/backend/batching differences rather than an isolated GDN control.

## Native and model admission

All 16 native cases pass: both K2560/N12288 and K4096/N2560 geometries at
rows 1/17/129/1536/2304/4096/29184/32768. Quantized-reference maximum relative
L2 is 0.00776%; BF16-relative error reaches 3.79% on synthetic inputs. Zero
rows, untouched other rows and changed-input CUDA-graph replay pass. Sampled
actual model weight reconstruction errors are 2.54–2.69%. All 48 projections
select native CUTLASS and execute. Retained 16 attention and 64 MLP scopes,
classifier/core/adapter identities and native receipts pass. FP4 preparation
and tile execution audits now cover only retained MLPs; removed FP4 GDN
producer calls are not required or mislabeled as executed.

The twenty-row canary has MAE/correlation **0.021057/0.992834** against the
selected default, failing its 0.005/0.995 agreement limits. Against the BF16
master, **0.014448/0.997776** passes the original 0.02/0.99 limits. All scores
and margins are finite, token counts exact and adapter effect nonzero.
Continue only as the requested finite precision diagnostic; this neither
relabels default agreement nor establishes held-out quality parity.

## Development benchmark

Warmups excluded, all repeats retained, fresh HTTP pools, frozen 320-row
1,310,581-token workload. Medians:

| GDN precision | c1 input tok/s | c1 req/s | c1 p50/p95, ms | c128 input tok/s | c128 req/s | c128 p50/p95, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected FP4 | 93,877 | 22.30 | 33.34 / 124.75 | 210,281 | 51.34 | 2.311 / 2.751 |
| FP8 | 96,305 | 22.88 | 29.19 / 134.72 | 188,825 | 46.10 | 2.587 / 3.062 |

c128 input throughput changes -10.20%; candidate range 180,255–189,502 versus
200,626–211,291 for the archived default. c1 throughput improves 2.59% with
better median but worse p95 latency. Sequential conditions do not establish
a precise isolated speed difference.

| GDN precision | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| Selected FP4 | 0.869013 | 0.892600 | 0.779613 | 0.613807 |
| FP8 | 0.869331 | 0.890002 | 0.733105 | 0.600118 |

Macro/pooled pAUROC changes -4.6507/-1.3689 points. Macro AUROC changes
+0.0318 points, pooled -0.2598. The candidate fails the predeclared development
gain/no-pooled-decline rule. Preserve 29 dual-label groups and the undefined
single-label Nemotron group; small groups and partial training overlap make
this systems cohort unsuitable for held-out parity claims. Mean/max score
drift is 0.05336/0.55774 with 18 fixed-0.5 flips. All per-source and repeat
metrics, calibration and tie diagnostics remain in the comparison artifact.

## Canonical ID

All 3,012 CoT-removed examples complete: 946 STRIDE, 2,066 Gloom and
33,750,959 prompt tokens. Ordered identities, labels, prompt hashes and token
counts match the archived controls. Original ordered 128-row partitions at
c128 remain fixed. One pass; repeat variation and statistical uncertainty
are not measured, and ID is descriptive rather than a selection loop.

| GDN precision | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 | Gloom pAUROC@20 | STRIDE pAUROC@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected FP4 | 0.957476 | 0.951578 | 0.865432 | 0.839805 | 0.785899 | 0.944966 |
| FP8 | 0.958720 | 0.952470 | 0.868042 | 0.840989 | 0.786065 | 0.950020 |

Macro/pooled pAUROC gains are +0.2610/+0.1183 points. Gloom gains only
0.0167 points, STRIDE 0.5054. The remaining macro gap to the archived
full-BF16 0.886090 result is 1.8047 points. Gloom remains about 3.93 points
below its full-BF16 result, so changing these projections to FP8 does not
recover that loss in this stack.

| GDN precision | ID seconds | Input tok/s | Req/s | p50/p95, s | Pooled Brier | Pooled ECE | Recall/FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected FP4 | 158.758 | 212,594 | 18.97 | 2.454 / 7.909 | 0.11657 | 0.13088 | 72.45% / 2.14% |
| FP8 | 176.982 | 190,703 | 17.02 | 2.831 / 8.894 | 0.09733 | 0.08667 | 80.29% / 3.92% |

Observed ID input throughput changes -10.30%. Mean score shifts +0.04421,
MAE is 0.05395 and maximum drift 0.53215 against the selected default. There
are 171 threshold flips: 161 upward, ten downward. Calibration and recall
improve while false alarms increase. Score resolution is 89 exact values
versus 90 for the selected default; preserve ties rather than attributing
the ranking differences to calibration alone.

## Evidence and closure

`native01` and complete `gdn02/fp8` live under `results/b200_gdn_precision/`.
`gdn01` preserves a failed client receipt-schema attempt: the model completed
its canary, but the shared guard required top-level correlation/MAE fields.
No timed passes ran in that attempt. The corrected driver reproduces its
canary scores exactly; guard tests cover finite failed parity, nonfinite scores
and zero adapter effect. Source snapshots and both attempts are retained.

Ninety-eight focused tests pass in the staged runtime; Ruff passes. Retain
14 Torch JIT deprecation warnings there and local NVML warnings. The completed
server log retains excluded startup/canary JIT warnings and intentional
shutdown `EngineDeadError`/one leaked-semaphore cleanup warning, after all
outputs were saved. Candidate API/engine retire successfully; closure records
no GPU processes and 0 MiB, with capacity, checkpoints and caches preserved.
All 102 collected files (238,440,250 bytes) verify by SHA256. Independent
tie-aware ROC integration verifies ID and development AUROC/pAUROC against
the saved metrics to 1e-12, with maximum discrepancy 2.22e-16;
`local_verification.json` records the checks.
