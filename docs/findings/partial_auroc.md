# One strict-filtered partial-AUROC variant

The single predeclared partial-AUC variant improves source-macro OOD normalized
pAUROC@20 by **0.7918 percentage points**, but lowers ID by **2.4025 points**.
It does not satisfy the hypothesis of improving ID with OOD transfer. Preserve
the strict soft-distillation baseline; no training-default promotion occurred.

## Intervention and reference

The [frozen experiment](../../experiments/partial_auroc/README.md) adds weight
0.1 to a source-macro pairwise softplus term comparing harmful samples with the
highest-scoring 20% of benign samples in each logical batch. Fractional tail
mass shares boundary ties equally; temperature is 1. Single-class sources retain
soft BCE only. This is a minibatch surrogate with small source-specific tail
pools, not direct optimization of the full evaluation pAUROC.

Both candidate and reference use the same 7,836 strictly filtered Kimi K3 rows,
soft targets, original hard labels, zero-B initialization, seed 0, rank 128,
learning rate 5e-5, one epoch and 245 updates. BF16/SDPA training preserves FP32
master adapters. The existing strict checkpoint is reused as the reference;
no additional baseline was trained. Evaluation uses merged BF16 backbones and
the validated FP32 final two-row head for both models, with archived
`fp32-monitor-head01` predictions as the matched reference. All 3,012 canonical
ID and 6,395 OOD examples and their rendered prompts/token counts match.

Only the final epoch was evaluated. No hyperparameter sweep, early stopping,
OOD tuning or checkpoint selection occurred. The OOD suite has prior repeated
use, and one seed does not establish robust improvement. Data and optimizer
updates match, but compute does not: logical ranking-gradient replay adds a
no-gradient score pass over another 75,851,675 tokens, on top of the same
75,851,675 gradient-pass tokens. The additional score passes take 1,032.90 s.

## Complete results

All values below are source-macro. Threshold diagnostics use 0.5.

| Population / objective | pAUROC@20 | AUROC | Brier | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| ID / strict soft BCE | 0.896106 | 0.963869 | 0.088830 | 21.02% | 94.17% |
| ID / + partial-AUC | 0.872081 | 0.951663 | 0.085184 | 14.74% | 90.48% |
| OOD / strict soft BCE | 0.787535 | 0.922867 | 0.136385 | 7.45% | 71.63% |
| OOD / + partial-AUC | 0.795452 | 0.923795 | 0.141138 | 4.93% | 69.44% |

Lower fixed-threshold false alarms coexist with lower recall on both populations.
ID Brier improves, while OOD Brier worsens. The ranking gain is heterogeneous:

| Source | Reference pAUROC@20 | Candidate | Change (percentage points) |
|---|---:|---:|---:|
| ID Gloom | 0.828650 | 0.770384 | -5.8266 |
| ID Stride | 0.963563 | 0.973778 | +1.0215 |
| OOD AgentDojo | 0.931627 | 0.940747 | +0.9120 |
| OOD CUA Shade Arena | 0.771250 | 0.775149 | +0.3899 |
| OOD IAC | 0.814451 | 0.859223 | +4.4773 |
| OOD MLE sabotage | 1.000000 | 1.000000 | 0.0000 |
| OOD Rogue Eval | 0.970485 | 0.979012 | +0.8527 |
| OOD Shade Arena | 0.237396 | 0.218584 | -1.8812 |

The ID regression is concentrated in Gloom; IAC contributes most to the OOD
gain. This is a measurable tail-ranking tradeoff, not a general improvement.
The receipts retain pooled metrics, source calibration, score ties and timing.
Candidate unique scores are 3,011 ID and 6,334 OOD. Scoring takes 294.74 s ID
and 493.15 s OOD, with 114,512 and 112,230 input tokens/s respectively; these
single-pass timings do not establish a throughput change.

## Execution and verification

`strict-pauc02` completed at **01:51:07 CEST on 2026-10-12** on the existing NC2
B200. Implementation is `d6b1637`; 39 focused tests and lint passed before launch.
Training completed all 245 updates, with finite gradient norms, exact complete
row/token coverage and zero maximum score/replay margin discrepancy. The fresh
parameter-gradient canary passed with maximum error `5.551115123125783e-17`.
Master/merge/serving parity passes: serving MAE 0.00180417, correlation
0.99994612; the comparison with the merged BF16 reference also passes. All eight
native BF16 attention calls and actual FP32 head dispatch are verified.

Collected outputs are under `results/strict-pauc02/`, logs under
`logs/runpod/strict-pauc02/`. `completion_audit.json` independently checks raw
responses, exact matched identities, full candidate/reference metric
recomputation, collected stage-product hashes, gradient replay and coverage,
and dtype/parity receipts. Prediction SHA256 values are:

- ID: `970a2ae17e482d67879cc357d056b7cfc9f726185a70b7cf549988bd09be6b8b`
- OOD: `6f339ed099c79fbf39fd6f4ec7cb00c94841577a457e5802cec787f1d3940d98`

The earlier `strict-pauc01` failed at offline tokenizer lookup before model
loading or any update. Its preserved failure receipt and the cache/PATH fix
are recorded in the experiment record; it was not a different objective trial.
Expected compilation specializations, PEFT config/exclusion warnings and the
experimental cuBLAS preference warning remain in logs. Scorer shutdown reports
one semaphore cleanup warning after complete predictions; no rows failed or
were excluded. The owned driver/scorer processes retired, and a subsequent
live GPU check shows 0% utilization and 0 MiB allocated. The Pod remains running,
with persistent adapters/caches and only the candidate in the shared disposable
merged-model slot; the previous strict master/export remain preserved.
