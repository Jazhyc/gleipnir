# Retain the baseline after the regularization screen

Date: 2026-08-14. Status: recorded in the decision log.

- All sixteen rank-128 screen cells completed on the fixed internal development
  split. The AdamW baseline remained best in macro AUROC at 0.96449; every
  warmup, schedule, horizon, dropout, weight-decay, sampling, and target-scale
  intervention reduced the primary metric.
- The frozen constraints and AUROC ranking retain square-root-balanced sampling,
  teacher-logit scale 1.5, weight decay 0.10, cosine scheduling, and
  teacher-logit scale 2.0 for possible seed-1/2 replication. These cells are
  trade-offs rather than wins. Their AUROC regressions range from 0.00025 to
  0.00198, although all improve balanced accuracy and four improve Brier.
- Do not combine or promote interventions from this single-seed screen. Retain
  the existing AdamW `5e-5`, linear, 3%-warmup, two-epoch, zero-dropout and
  zero-weight-decay, proportional-sampling, unscaled-target recipe unless the
  frozen shortlist wins replicated evaluation. Competition validation and the
  final test remained untouched.
- Teacher-logit scale 1.5 is the most interesting operating-point trade-off:
  relative to baseline it changes macro AUROC by -0.00091, balanced accuracy by
  +0.01000, and Brier by -0.00142. Scale 2.0 is dominated by scale 1.5 on all
  three macro metrics and creates substantially more tied scores, so it should
  not be preferred merely because it passed the formal shortlist rule.
