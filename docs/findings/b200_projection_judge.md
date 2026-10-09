# APPS projection transfer to JudgeDeceiver

2026-10-09. The same full projection that nearly eliminates APPS honest-injection
alarms has **little aggregate benefit in JudgeDeceiver preference judging**.
Preferred-injected accuracy improves only 0.65 points, with opposing source
effects. Pooled ranking improves slightly while calibration worsens. The
[experiment contract](../../experiments/b200_projection_judge/README.md) owns
the frozen transfer, gates, populations and stop conditions; the
[APPS finding](b200_injection_direction.md) owns fitting and original effects.

## Fixed transfer and matched control

Transfer the already fitted unit layer-20 direction unchanged: full projection
at all 32 layers/all tokens, beta 1, zero centers. No JudgeDeceiver fitting,
strength/layer/threshold sweep, retraining or promotion. Use the current
SDPA augmented master `bdb702316b1d...` and checksum-identical merged BF16
checkpoint, vLLM 0.31/Torch 2.13/Lens 1.3.0 eager research serving on the same
user-reserved NC2 B200.

Reuse the exact original 4,188 A/B prompts: 504 clean, 1,842 preferred-injected
and 1,842 disfavored-injected, 252 pairs, both orders, six held-out query groups.
All preferences and prompt/token identities remain frozen; ties at p(B)=.5
choose B. These labels are preferences, not harmfulness or injection FPR.
Source revision, construction labels, unresolved upstream license and the
dependent order/suffix grouping follow the
[original JudgeDeceiver contract](../../experiments/judge_injection_continuation/README.md).
The current SDPA adapter did not train on this dataset. Previous FA4/historical
adapter scores do not substitute for this fresh unedited SDPA control.

## Results

| Metric | Unedited SDPA | Full projection |
|---|---:|---:|
| Clean accuracy | 97.82% | 99.01% |
| Preferred-injected accuracy | 55.81% | 56.46% |
| Disfavored-injected accuracy | 98.05% | 98.37% |
| Pooled accuracy | 79.44% | 80.01% |
| Pooled AUROC | 0.910254 | 0.914715 |
| Pooled raw pAUROC20 | 0.686680 | 0.696586 |
| Preferred-injected AUROC | 0.631926 | 0.638094 |
| Preferred-injected raw pAUROC20 | 0.233552 | 0.224667 |
| Source-macro accuracy / AUROC | 80.46% / 0.916053 | 80.57% / 0.931113 |
| Query-macro accuracy / AUROC | 80.11% / 0.899474 | 80.38% / 0.912004 |
| Pooled Brier / log loss | 0.127261 / 0.375821 | 0.130899 / 0.387435 |
| Pooled ECE | 0.091809 | 0.112615 |
| Mean p(correct) | 0.751156 | 0.749889 |
| Exact-half ties | 106 | 45 |

The 182 decision flips comprise 103 correct decisions gained and 79 lost:
clean gains six/loses zero; preferred injections gain 86/lose 74; disfavored
injections gain 11/lose five. Preferred-injected score MAE is 0.054956, despite
only twelve net correct decisions gained. Its mean p(correct) decreases
0.555634 to 0.552233, and raw p20 decreases; the accuracy gain is not a uniform
ranking/confidence improvement.

| Preferred-injected source | Rows | Unedited accuracy | Projection accuracy | Change |
|---|---:|---:|---:|---:|
| LLMBar | 888 | 61.04% | 56.87% | -4.17 pp |
| MT-Bench | 896 | 50.33% | 56.25% | +5.92 pp |
| RLAIF | 20 | 60.00% | 50.00% | -10.00 pp |
| Search | 38 | 60.53% | 55.26% | -5.26 pp |

Preferred clean-to-injection mean p(correct) change is -0.371889 unedited versus
-0.375654 projected; correct-to-wrong injection flip rates are 43.21% versus
42.62%. Disfavored score shifts are -0.023765 versus -0.024403, with flip rates
1.52% versus 1.19%. Denominators retain every paired injection variant.
The full six-query accuracy breakdown is in `summary.json`; query-macro
accuracy changes by only 0.26 points. Six queries, few rare-source rows,
dependent variants and one model/run do not establish broad or seed-robust
transfer. No independent-rows significance claim is made.

Both arms retain A/B probability mass: means 0.966640/0.968585, minima
0.775898/0.766040; no row below .5. These are readout diagnostics, not generation
format tests. The direction's large APPS effect does not transfer reliably to
this other prompt/task/readout. This experiment cannot isolate which of those
differences limits transfer.

## Readout validation and receipts

The Lens full-vocabulary path now also returns A/B rows 32/33 and p(A or B)
from the actual fused final-normalization result and tied embedding. Its
resident cached 0/1 head/backbone remain identical. A separate FP32 selected-row
projection of the same normalized hidden validates the head. The original
30 training-derived canaries pass MAE **0.004628**, correlation **0.999804**
against that reference. This is a head check, not a new whole-backbone
FP32-master A/B parity comparison. Reuse the unchanged adapter/merge receipts;
fresh existing 0/1 master/native gates pass. The new eager 20-row no-op canary
reproduces the previous engine exactly. Real A/B beta-zero/effect/next-plain
restoration checks pass. Focused runtime tests pass **19**, with 14 inherited
Torch deprecation warnings; local preference contract tests pass **two**.
Scoped Ruff and diff checks pass.

`results/b200_projection_judge/sdpa_judge_projection01/` contains the frozen
manifest/config/workload, executed sources, readout gate/smoke, per-batch scores
and timing, both prediction files, summary and completion status. All **99
collected files** (23,446,021 bytes) pass collection checksums. The independent
audit binds all **8,376 held-out predictions** to canonical labels/prompts,
recomputes logits/sigmoid, condition/source/query AUROC/raw p20 and accuracy,
paired injection effects, gained/lost decisions and token/timing totals.
Scripts and logs are in `logs/local/b200_projection_judge/`.

Startup/retirement evidence uses `results/b200_sdpa_lens/sdpa03_judge_projection/`
and `results/b200_vllm031/sdpa02_before_judge_projection_retired.json`.
The passing engine stays warm, API 3883/worker 3935, with zero capture/steering/
projection entries and zero uncorrected volatile ECC errors. Weights remain
on the B200; no capacity, dependency or training-runtime changes.

## Single-pass end-to-end costs

Concurrency 32, batches of 128, prefix cache off, final-layer capture and
full-vocabulary/A/B readouts on every row. Each arm scores **1,521,530 input
tokens**. Timing includes batch wall time; request latency includes semaphore
wait. These are diagnostic campaign costs, not repeated production benchmarks.

| Arm | Seconds | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|---:|
| Unedited | 66.93 | 22,732 | 62.57 | 1.07 / 2.08 |
| Projection | 98.94 | 15,378 | 42.33 | 1.59 / 2.96 |
