# Compare Muon through scheduled update-RMS matching

Date: 2026-08-13.
Status: historical; governs no future work.

- Replace the independent, unscheduled Muon learning rate with the standard
  scheduler-managed parameter-group `lr`. Apply the per-matrix Moonshot scaling
  `0.2 * sqrt(max(A, B))`, so AdamW and Muon can be compared at the same five
  base learning rates despite LoRA A/B shape differences.
- Screen paired AdamW and Muon rates on rank 64, 25% training data, and a fixed
  internal dataset-label-stratified development split. Do not use competition
  validation for hyperparameter selection. The source artifact has no available
  conversation/generator lineage key, so record that the internal split cannot
  enforce lineage grouping.
- Keep the linear schedule, 3% warmup, two epochs, QLoRA/FLA path, batch, seed,
  data, and objective fixed. Treat the ten-cell screen as pruning evidence and
  require replicated confirmation before promotion.
