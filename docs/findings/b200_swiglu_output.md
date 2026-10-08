# Direct FP4 SwiGLU-output ablation on the FP8-attention default

2026-10-08. Disabling direct FP4 SwiGLU output improves development ranking but
**worsens full ID source-macro pAUROC@20 by 0.913 percentage points**. Both
Gloom and STRIDE regress. Input throughput changes less than 1% in the median
development benchmark and the single ID pass. This ablation does not recover
the remaining ID gap within the current stack; retain the selected default.
It leaves SiLU/packing and normalization fusion enabled, so those interventions
and the attention core remain separate questions.

## Frozen intervention

The [experiment contract](../../experiments/b200_swiglu_output/README.md)
predeclares one candidate, the matched development benchmark and one complete
ID pass regardless of development results. The control is the saved
`b200_attention_precision/precision01/fp8` default: FP8 attention projections,
FP4 GDN and MLP weights, native direct FP4 SwiGLU output and MXFP8 attention.
Reuse its predictions and timing repeats on the same B200 UUID, without rerun.

Only the large-row activation producer changes. Above 4,096 physical rows,
replace direct local-block FP4 emission with the existing symbolic-row fused
GEMM/SwiGLU BF16 output and separate whole-row dynamic FP4 packing. The down
projection still consumes FP4 activations and weights. Retain the same small
and medium-row arithmetic, fusion, normalization, final 272-update FP4/FA4
adapter, merged model, vLLM 0.31 runtime, native frontend/direct FROST,
CUDA GDN/automatic context parallelism, stock synchronous FCFS, prefix caching
off and 32K/128 limits. No OOD, checkpoint/threshold fitting or automatic
promotion. The archived full-BF16 result remains a whole-stack comparison.

## Native and model admission

Refresh nine native cases at rows 1/17/129/1536/1537/2304/4096/29184/32768.
All pass the unchanged 1% reference ceiling, exact zero/untouched rows and
nonzero changed-input graph replay with one symbolic-row plan. Down-projection
outputs and changed-replay references are exact; maximum BF16 activation
relative L2 is 1.57e-11. The model audits observe all 32 fused whole-row MLPs
and **zero direct-output calls**. Unexpected direct dispatch fails closed.
All 16 FP8 attention, 48 FP4 GDN and 64 FP4 MLP scopes, classifier and inherited
core/native/adapter identities remain intact.

The 20-row canary is finite with exact tokens and nonzero adapter effect.
MAE/correlation against the default is **0.005277/0.999496**, narrowly failing
the 0.005 MAE limit. Against the BF16 master it is **0.025677/0.992790**,
failing the 0.02 MAE limit. Preserve both failures under the user-requested
finite diagnostic; native correctness does not establish score/quality parity.

## Development benchmark

Three quick64/c1 and six full320/c128 repeats, excluded warmups, fresh HTTP
pools and all observations retained. Frozen full workload: 320 examples and
1,310,581 prompt tokens. Medians:

| SwiGLU output | c1 input tok/s | c1 req/s | c1 p50/p95, ms | c128 input tok/s | c128 req/s | c128 p50/p95, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct FP4 default | 93,877 | 22.30 | 33.34 / 124.75 | 210,281 | 51.34 | 2.311 / 2.751 |
| BF16 + whole-row FP4 pack | 95,322 | 22.64 | 32.79 / 123.37 | 208,749 | 50.97 | 2.353 / 2.777 |

Median c128 throughput changes -0.73%, with candidate range 174,497–209,607
against the default's 200,626–211,291. The first timed c128 pass includes a
`fused_sigmoid_gating_delta_rule_update_kernel` JIT warning after 859 completed
score requests: after the canary, four quick64 passes and full320 warmup, within
the first full timed pass. Preserve it rather than discarding the slower result.
c1 input throughput changes +1.54%. Sequential timing and overlapping ranges
do not establish an isolated sub-percent kernel cost.

| SwiGLU output | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| Direct FP4 default | 0.869013 | 0.892600 | 0.779613 | 0.613807 |
| Whole-row pack | 0.888327 | 0.907505 | 0.785536 | 0.649832 |

Macro/pooled AUROC gains are +1.9314/+1.4905 percentage points; pAUROC gains
+0.5923/+3.6025. The candidate meets the development diagnostic preference.
This mixed systems cohort has partial training overlap, 29 dual-label groups
and one undefined single-label Nemotron group; it is not held-out quality
evidence. Keep every per-source/repeat metric. Mean/max score drift is
0.04909/0.47441, with 18 fixed-0.5 threshold flips.

## Canonical ID

Complete all 3,012 CoT-removed examples: 946 STRIDE, 2,066 Gloom and
33,750,959 prompt tokens. Ordered identities, labels, prompt hashes and exact
token counts match the controls. Retain the original ordered 128-row partitions
at c128, excluded warmup and one descriptive pass. Repeat variation and
statistical uncertainty are unmeasured; there is no ID-driven search loop.

| SwiGLU output | Macro AUROC | Pooled AUROC | Macro pAUROC@20 | Pooled pAUROC@20 | Gloom pAUROC@20 | STRIDE pAUROC@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct FP4 default | 0.957476 | 0.951578 | 0.865432 | 0.839805 | 0.785899 | 0.944966 |
| Whole-row pack | 0.955209 | 0.948938 | 0.856300 | 0.829911 | 0.773192 | 0.939408 |

Macro/pooled pAUROC declines 0.9132/0.9895 points. Gloom declines 1.2706,
STRIDE 0.5558. Macro/pooled AUROC declines 0.2267/0.2641 points. The native
producer checks and development gains do not predict this ID result.

| SwiGLU output | ID seconds | Input tok/s | Req/s | p50/p95, s | Pooled Brier | Pooled ECE | Recall/FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct FP4 default | 158.758 | 212,594 | 18.97 | 2.454 / 7.909 | 0.11657 | 0.13088 | 72.45% / 2.14% |
| Whole-row pack | 160.169 | 210,721 | 18.81 | 2.558 / 8.094 | 0.12113 | 0.13709 | 71.77% / 2.42% |

Observed ID input throughput changes -0.88%. Mean score shifts -0.00620,
MAE is 0.03126 and maximum drift 0.41157. There are 83 threshold flips,
38 upward and 45 downward. Exact unique scores change 90 to 89. Preserve
calibration, ties and all source/length diagnostics alongside ranking metrics.

## Evidence and closure

`results/b200_swiglu_output/output01/whole_row` holds the complete campaign,
source snapshots, settings, canary, development repeats, ID predictions,
comparisons and retirement. `native01` is the executed native gate. After
collection, fix a CPU-only fixture's missing capture-state mock and make the
diagnostic CLI exit nonzero on failed native checks. `native02` repeats all
nine passing cases for the final CLI bytes; these changes do not alter kernel
arithmetic or rerun ID. Original campaign source/config snapshots remain intact.

Eleven focused tests pass in the final staged runtime and the corrected worker
fixture passes locally; Ruff passes. Preserve the original CPU-only fixture
failure, 14 Torch JIT deprecations and local NVML warnings. Retain startup JIT
warnings, the included timed GDN JIT spike, and intentional-shutdown
`EngineDeadError`/one leaked-semaphore cleanup warnings after complete outputs.
The owned API/engine retire successfully; closure records no GPU processes
and 0 MiB. Capacity, model/master weights and persistent caches remain intact.
All 104 collected files (121,500,512 bytes) verify by SHA256. Independent
tie-aware ID/development ROC integration matches AUROC/pAUROC to 1e-12,
maximum discrepancy 2.22e-16, recorded in `local_verification.json`.
