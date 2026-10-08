# Full-attention projection precision recovery

2026-10-08. On the fixed FP4-trained 4B adapter, FP8 full-attention projections
recover 1.16 percentage points of ID source-macro pAUROC@20 over the 0.31 FP4
recipe; BF16 projections recover 1.02 points. FP8 is slightly faster than BF16
and slightly better on this single ID observation. Both leave a substantial
gap from the same adapter's archived full BF16 serving result. The frozen
development triage favors BF16, while ID is descriptive only; no reference or
production precision is promoted.

## Frozen intervention

The [experiment contract](../../experiments/b200_attention_precision/README.md)
predeclares both candidates, development triage and one ID pass for each,
regardless of development results. Only the eight full-attention blocks'
16 QKV/output projections change. Both load the original merged BF16 weights;
FP8 uses E4M3 per-output-channel weights, dynamic per-token activations, native
CUTLASS GEMM and BF16 output. BF16 uses an opaque native BF16 linear call.
Actual-call auditing occurs inside these opaque operators, outside compilation
and CUDA graph capture. Preserve FP4 MLP/GDN projections and native SwiGLU
output, MXFP8 attention core, BF16 cache/recurrence, FP32 gates/state, CUDA GDN
with automatic context parallelism, native frontend/direct host bindings,
two-row BF16 classifier, synchronous stock FCFS, prefix caching off and 32K
context/chunk/128-sequence limits.

The adapter is the final 272-update FP4/FA4 training replication: FP32 causal
master `8dbc1a2e...`, serving export `d13be8b2...`, same checksum-bound merged
checkpoint. Reuse archived 0.31 FP4 development repeats at
`b200_vllm031/diagnostic_default02` and ID at `b200_vllm031/id02`; no control
rerun. Retain 0.24 optimized and full BF16 same-adapter ID predictions as
additional controls. The latter is a historical dynamic-LoRA whole-stack
comparison on another host, not an isolated projection-precision control.

## Native and loaded validation

All 32 BF16/FP8 native cases pass: both projection geometries, rows
1/17/129/1536/2304/4096/29184/32768, decoded FP32 arithmetic reference,
zero rows, unchanged other rows and changed-input CUDA-graph replay. Maximum
FP8 output error versus BF16 is 3.87% on these synthetic inputs, compared with
about 14.6% in the earlier FP4 operator screen (different saved fixtures).
Actual model FP8 weight reconstruction error on sampled columns is
2.58–2.69%; every target projection selects native W8A8 and executes. Both
loaded models retain all 64 FP4 MLP and 48 FP4 GDN projections and pass the
unchanged native/head/core/adapter identity audits.

Canaries remain separately reported against each reference:

| Projection precision | MAE vs accepted 0.24 | MAE vs BF16 master | Master strict pass |
| --- | ---: | ---: | --- |
| FP4 0.31 control | 0.020600 | 0.027090 | No |
| BF16 | 0.015197 | 0.015927 | Yes |
| FP8 | 0.015596 | 0.020901 | No |

All are finite with exact tokens and nonzero adapter effects. Neither candidate
meets the accepted-0.24 MAE 0.005 guard. FP8 narrowly misses the master MAE 0.02
guard; correlation exceeds 0.99. Preserve these failures under the user's
explicit finite precision-recovery diagnostic authorization.

## Development benchmark

Three quick64/c1 and six full320/c128 repeats per candidate, excluded warmups,
fresh HTTP pools and every timed pass retained. Same B200 UUID and frozen
1,310,581-token full workload. Medians:

| Projection precision | c1 input tok/s | c1 req/s | c1 p50/p95, ms | c128 input tok/s | c128 req/s | c128 p50/p95, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived FP4 | 94,980 | 22.56 | 33.47 / 120.44 | 218,132 | 53.26 | 2.223 / 2.596 |
| BF16 | 93,646 | 22.25 | 33.27 / 125.42 | 206,302 | 50.37 | 2.368 / 2.798 |
| FP8 | 93,877 | 22.30 | 33.34 / 124.75 | 210,281 | 51.34 | 2.311 / 2.751 |

BF16/FP8 c128 throughput changes -5.42%/-3.60%; FP8 is 1.93% above BF16.
Ranges are 195,726–206,920 BF16 and 200,626–211,291 FP8; each retains a slower
pass. Sequential conditions do not establish a precise causal speed difference.

| Projection precision | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| FP4 | 0.876953 | 0.882267 | 0.771184 | 0.571076 |
| BF16 | 0.878878 | 0.898910 | 0.790101 | 0.625623 |
| FP8 | 0.869013 | 0.892600 | 0.779613 | 0.613807 |

Macro/pooled pAUROC gains are +1.8917/+5.4547 points for BF16 and
+0.8429/+4.2731 for FP8. FP8 macro AUROC regresses 0.7939 points despite better
partial AUC. Preserve all per-source metrics: 29 dual-label groups and one
undefined single-label Nemotron group, including small four/six-row sources.
The mixed cohort has only 130 identities in this adapter's training data and
cannot establish held-out quality parity. Both meet the frozen development
gain/no-pooled-decline rule, but FP8 is 1.0488 pAUROC points below BF16, outside
the declared 0.5-point proximity window. Thus the diagnostic triage prefers
BF16; it does not use ID or promote a recipe.

## Canonical ID

Both complete all 3,012 CoT-removed examples: 946 STRIDE and 2,066 Gloom,
33,750,959 prompt tokens. Ordered identities, labels, source/rendered prompt
hashes and token counts match both archived controls. Original 128-row
partitions and c128 admission remain fixed; one pass each, no repeat-variation
claim, threshold fitting, checkpoint selection or OOD use.

| Projection precision | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 | Gloom pAUROC@20 | STRIDE pAUROC@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP4 0.31 | 0.954522 | 0.948902 | 0.853837 | 0.827293 | 0.761728 | 0.945945 |
| BF16 projections | 0.956248 | 0.950122 | 0.864082 | 0.837939 | 0.785026 | 0.943139 |
| FP8 projections | 0.957476 | 0.951578 | 0.865432 | 0.839805 | 0.785899 | 0.944966 |
| Archived full BF16 stack | 0.965927 | 0.960818 | 0.886090 | 0.866566 | 0.825344 | 0.946835 |

FP8/BF16 macro pAUROC gains over FP4 are 1.1596/1.0246 points, recovering
approximately 36%/32% of the observed FP4-to-full-BF16 gap. Gloom gains
2.4170/2.3298 points, while STRIDE regresses 0.0979/0.2806 points. FP8's
0.1350-point advantage over BF16 is a single-pass observation and does not
establish a robust ordering. The remaining gap requires separate interventions;
these results do not identify GDN, attention-core or MLP precision as its cause.

| Projection precision | ID seconds | Input tok/s | Req/s | p50/p95, s | Pooled Brier | Pooled ECE | Recall/FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP4 | 154.102 | 219,017 | 19.55 | 2.444 / 7.765 | 0.13720 | 0.16485 | 65.61% / 1.28% |
| BF16 | 162.906 | 207,181 | 18.49 | 2.561 / 8.222 | 0.11824 | 0.13423 | 72.39% / 2.14% |
| FP8 | 158.758 | 212,594 | 18.97 | 2.454 / 7.909 | 0.11657 | 0.13088 | 72.45% / 2.14% |

Scores shift upward versus FP4: mean +0.03062 BF16/+0.03396 FP8, MAE
0.05138/0.05267 and maximum 0.54621/0.59040. There are 169/174 threshold
flips, mostly upward. Better recall accompanies more false alarms; calibration
improves but neither candidate reproduces the archived scores exactly.

## Evidence and closure

Artifacts live in `results/b200_attention_precision/precision01/` and logs in
`logs/runpod/b200_attention_precision/`. `comparison_summary.json` contains
all timing/quality/score summaries and frozen development triage. All 145
collected files (124,725,229 bytes) verify by SHA-256. Independent local ROC
integration and score/margin checks reproduce both ID conditions to 1e-12.
Source/config snapshots preserve the measured revision; a subsequent portable
command-test fixture cleanup changes only future test-source bindings.

Final focused tests pass 37 cases in the actual staged 0.31 runtime; repository
Ruff passes. Local checks also pass (35 cases before fixture extensions), with
slow shared-filesystem imports and NVML unavailable. Retain Torch JIT
deprecation warnings, excluded-canary/warmup JIT notices and intentional
shutdown EngineDeadError/resource-tracker warnings. Startup is excluded:
148.417 seconds BF16 and 156.364 seconds FP8. Both servers retire with no GPU
processes and 0 MiB allocated. Capacity, master/merged weights, shared compiler
caches, failed receipts and all frozen baselines remain intact.
