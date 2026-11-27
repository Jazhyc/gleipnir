# Do not promote hard-only from the internal split

Date: 2026-08-17. Status: recorded in the decision log.

- The unadapted Qwen3.5-9B base reaches 0.92273 macro AUROC on the exact 1,972-row
  internal development surface (0.88903 pooled AUROC). Hard-only adaptation
  reaches a three-seed mean of 0.97256 macro AUROC, but its family gains are
  highly uneven: +0.00644 on instructed Aletheia, +0.05385 on varied Aletheia,
  and +0.17097 on Liars Bench.
- Interpret the hard-only result as strong within-source adaptation, not evidence
  of deployment-distribution transfer. The development split is stratified by
  dataset and label but cannot group unavailable conversation/generator lineage,
  and the same dataset families occur in train and development.
- Retain the mixed soft-plus-hard weight-0.50 recipe as the internal AUROC
  candidate because it is stronger on the varied family (0.95078 versus 0.93589
  for hard-only). Freeze recipes now; compare the base, hard-only, and mixed
  candidate once on an untouched source-grouped surface before promotion.
