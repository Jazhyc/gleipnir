# Screen trajectory, regularization, and data weighting

Date: 2026-08-14. Status: recorded in the decision log.

- After Muon failed full-data confirmation, keep AdamW and screen sixteen
  rank-128 cells: one shared baseline plus fifteen single-factor changes to
  warmup, schedule, epoch count, LoRA dropout, weight decay, teacher-logit
  smoothing, and dataset sampling.
- Reserve a new fixed 15% internal development split, train every cell on the
  same 50% selection from its complement, and keep competition validation and
  final test untouched. Treat the single-seed screen as pruning evidence only;
  replicate at most five eligible cells before combining interventions.
- Add deterministic square-root-balanced and uniform-dataset sampling. Record
  expected source exposure because source sizes differ by about 15x while the
  primary metric macro-averages datasets. Retain causal model-only checkpoints
  about every half epoch and record loss/LR history so horizon effects remain
  inspectable without optimizer-state storage.
