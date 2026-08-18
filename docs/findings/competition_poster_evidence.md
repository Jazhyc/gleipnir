# Competition poster evidence

This document freezes the completed deception-monitor results available for a
competition poster as of 2026-08-18. It is a reporting record, not a new model
selection surface. Generated predictions and full summaries remain in ignored
`results/` trees; the tables below record the durable numbers and qualifications.

## Evaluation surfaces

Do not compare absolute metrics across these three surfaces as if they were the
same test set:

1. **Competition validation:** 822 rows and 21 datasets. Both data-scaling
   experiments use these exact rows, but use different prompts/readout protocols
   and different training pools.
2. **Internal objective development:** 1,972 rows drawn from the same broad
   source families as the 11,177-row training complement. This repeatedly
   inspected surface selected the hard-label strength and is not OOD evidence.
3. **AISI external transfer:** 49,685 rows across 45 subject-model/testbed groups.
   It contains Varied Deception, AuditBench, Gender Secret, and Sandbagging and
   was not used to fit the adapters.

The final competition test split remains untouched.

## Supervision data scaling

The two curves share the 822-row competition validation set, but **not the same
training fractions or row counts**. Kimi soft-logit distillation uses
`1/2/4/8/16/32/64/100%` of a 13,149-row pool and three seeds. Reasoning SFT uses
`1/2/5/10/20/40/80/100%` of a 2,877-row privileged-summary pool and one
historical seed. The former reads the direct decision-boundary logit; the latter
generates a reasoning prefix and reads the terminal decision logit. Plot them as
separate panels or against actual training rows, not as a matched fractional
comparison.

### Kimi K3 soft-logit distillation

| Fraction | Rows | Macro AUROC | Macro BA |
|---:|---:|---:|---:|
| Base | 0 | 0.94512 | 0.88452 |
| 1% | 149 | 0.94708 ± 0.00206 | 0.89563 ± 0.00182 |
| 2% | 256 | 0.94643 ± 0.00158 | 0.90040 ± 0.00727 |
| 4% | 511 | 0.94716 ± 0.00108 | 0.90675 ± 0.00383 |
| 8% | 1,060 | 0.94937 ± 0.00198 | 0.90754 ± 0.00069 |
| 16% | 2,120 | 0.95089 ± 0.00046 | 0.91032 ± 0.00248 |
| 32% | 4,199 | 0.95633 ± 0.00054 | 0.91706 ± 0.00300 |
| 64% | 8,398 | 0.95950 ± 0.00036 | 0.91984 ± 0.00300 |
| 100% | 13,149 | 0.96206 ± 0.00182 | 0.92222 ± 0.00137 |

The curve supplies a coherent positive data-scaling result, with most visible
AUROC separation appearing beyond 16% data. A bounded descriptive fit gives
macro-AUROC exponent `0.07734` and RMSE `0.00206`; its asymptote lands on the
upper bound `1.0`, so do not use it for confident long-range extrapolation.

### Privileged-reasoning SFT

| Fraction | Rows | Macro AUROC | Macro BA |
|---:|---:|---:|---:|
| Base | 0 | 0.93452 | 0.90714 |
| 1% | 36 | 0.92399 | 0.90357 |
| 2% | 54 | 0.93060 | 0.90595 |
| 5% | 144 | 0.92482 | 0.90357 |
| 10% | 288 | 0.93488 | 0.89881 |
| 20% | 576 | 0.91988 | 0.88929 |
| 40% | 1,152 | 0.92429 | 0.89762 |
| 80% | 2,301 | 0.92244 | 0.88214 |
| 100% | 2,877 | 0.92351 | 0.88095 |

The best point, 10%, exceeds the matched base AUROC by only `0.00036`, while the
full-data adapter is worse than base. Both AUROC and balanced-accuracy power-law
fits are rejected. Correct checksum-rebased adapters were used, so the flat
result is not the earlier silent key-mismatch failure. This supports the claim
that privileged-reasoning SFT saturates almost immediately and does not exhibit
the coherent scaling seen for direct soft-logit distillation.

## Supporting adapter-capacity result

The full-data QLoRA sweep separately varied rank with the Kimi soft-target
objective. Each point is a three-seed mean on the same 822-row competition
validation set.

| Rank | Macro AUROC | Macro BA |
|---:|---:|---:|
| 1 | 0.95192 ± 0.00024 | 0.92103 ± 0.00182 |
| 2 | 0.95496 ± 0.00164 | 0.92183 ± 0.00300 |
| 4 | 0.95698 ± 0.00061 | **0.92222 ± 0.00418** |
| 8 | 0.95821 ± 0.00202 | 0.92143 ± 0.00357 |
| 16 | 0.96081 ± 0.00068 | 0.92103 ± 0.00137 |
| 32 | 0.96149 ± 0.00147 | 0.91667 ± 0.00412 |
| 64 | 0.96254 ± 0.00130 | 0.91389 ± 0.00137 |
| 128 | **0.96409 ± 0.00216** | 0.91468 ± 0.00481 |
| 256 | 0.96333 ± 0.00495 | 0.91587 ± 0.00069 |

Macro AUROC scales with rank and then saturates: the bounded rank fit has
exponent `0.31121`, asymptote `0.96655`, and RMSE `0.00043`. The balanced-
accuracy fit is rejected and BA worsens at high rank, so the capacity result is
specific to ranking quality rather than a universal metric improvement.

A preregistered 4-by-4 data/rank surface tested ranks `4/16/64/256` at
`10/25/50/100%` data with three paired seeds. The mean rank-256 minus rank-16
AUROC contrast changes from `-0.00236`, through `-0.00077` and `+0.00079`, to
`+0.00252` as data increases. Its slope against log training rows is
`0.00212 ± 0.00122`. This is suggestive, but noisy, evidence that useful adapter
capacity increases with data; rank 64 is the observed winner through 50% and
rank 256 at 100%. It is a secondary poster result, not a precise scaling law.

## Internal supervision-objective comparison

All adapters below are rank-128 QLoRAs trained on the same 11,177-row complement
and evaluated on the 1,972-row internal development set. Adapter entries are
three-seed mean ± sample standard deviation; the unadapted base is deterministic.
Brier is lower-is-better.

| Objective | Macro AUROC | Macro BA | Macro Brier |
|---|---:|---:|---:|
| Unadapted base | 0.92273 | 0.85373 | 0.09790 |
| Soft-only | 0.96701 ± 0.00207 | 0.91128 ± 0.00396 | 0.07186 ± 0.00228 |
| Hard-only | 0.97256 ± 0.00092 | **0.93551 ± 0.00508** | 0.05543 ± 0.00206 |
| Soft + 0.50 hard | **0.97714 ± 0.00090** | 0.92645 ± 0.00456 | **0.05480 ± 0.00219** |

The mixed objective wins internal macro AUROC and Brier, while hard-only wins
balanced accuracy at the fixed threshold. These gains are partly source-specific:
hard-only is nearly saturated on instructed Aletheia data and gains much more on
Liars Bench than on varied Aletheia data. Treat the internal table as adaptation
evidence, not broad transfer evidence.

## External AISI transfer

The base is evaluated once; adapter entries are three-seed mean ± sample
standard deviation.

| Objective | Subject-macro AUROC | Subject-macro BA | Pooled AUROC | Pooled BA |
|---|---:|---:|---:|---:|
| Unadapted base | 0.84718 | 0.75531 | 0.92561 | 0.84893 |
| Soft-only | **0.85347 ± 0.00937** | **0.76640 ± 0.00456** | **0.95283 ± 0.00148** | **0.86452 ± 0.00935** |
| Hard-only | 0.83707 ± 0.00538 | 0.73436 ± 0.03902 | 0.92801 ± 0.01065 | 0.80985 ± 0.05755 |
| Soft + 0.50 hard | 0.85334 ± 0.00757 | 0.76333 ± 0.00427 | 0.94795 ± 0.00171 | 0.85312 ± 0.00801 |

| Objective | AuditBench AUROC | Gender Secret AUROC | Sandbagging AUROC | Varied Deception AUROC |
|---|---:|---:|---:|---:|
| Unadapted base | 0.54024 | **0.82011** | 0.61601 | 0.93215 |
| Soft-only | 0.52950 ± 0.01860 | 0.69057 ± 0.07641 | 0.59950 ± 0.02697 | **0.96057 ± 0.00085** |
| Hard-only | 0.54077 ± 0.02549 | 0.63981 ± 0.03490 | 0.57689 ± 0.05698 | 0.94280 ± 0.00603 |
| Soft + 0.50 hard | **0.54536 ± 0.00919** | 0.66615 ± 0.07005 | **0.62742 ± 0.03096** | 0.95823 ± 0.00127 |

Soft-only is the strongest adapter on pooled and Varied Deception metrics. Mixed
training is effectively tied on the 45-subject macro and is stronger on
AuditBench and Sandbagging. Hard-only does not transfer robustly. All adapted
models regress on Gender Secret, where the supplied transcript does not reveal
the hidden fact needed to infer the label.

As an explicit diagnostic, routing the four known Gender Secret subjects to the
base gives `0.86498 ± 0.00288` subject-macro AUROC with soft-only elsewhere and
`0.86702 ± 0.00143` with mixed elsewhere. This is evidence for an observability
problem, not a general solution for unknown deployment inputs.

## Poster-level conclusions

1. Direct soft-logit distillation exhibits coherent data scaling; generated
   privileged-reasoning SFT does not.
2. Hard labels produce large internal gains but do not consistently improve OOD
   transfer. Soft-only currently transfers best overall, while mixed supervision
   helps selected external testbeds.
3. More diverse, observable training environments are the most credible next
   scaling lever. More labels cannot solve settings where the student input does
   not identify the target.
4. These experiments support a focused deception-monitor poster. A broader
   multi-domain Gleipnir foundation model remains future work and should not be
   implied by the current evidence.

## Artifact pointers

- `results/data_scaling/summary/summary/scaling_curve.csv`
- `results/reasoning_sft_scaling/summary/scaling_curve.csv`
- `results/adapter_capacity_scaling_qlora/summary/summary/capacity_curve.csv`
- `results/data_capacity_scaling_qlora/summary/interaction_summary.json`
- `results/training_procedure_screen/hard_label_strength/summary/strength_curve.csv`
- `results/aisi_lie_detection_transfer/summary/summary.csv`

These paths are intentionally ignored by Git. Reconstruct them through the
experiment READMEs and their recorded source revisions/checksums.
