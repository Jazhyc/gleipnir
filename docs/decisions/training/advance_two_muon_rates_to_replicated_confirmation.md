# Advance two Muon rates to replicated confirmation

Date: 2026-08-13. Status: recorded in the decision log.

- The rank-64, 25%-data internal-development screen completed all ten paired
  cells. Muon `5e-5` won the predeclared primary metric with 0.94727 macro
  AUROC, +0.00312 over AdamW at the same rate. Muon `1e-4` was the strongest
  multi-metric alternative, with 0.94608 macro AUROC, 0.88942 macro balanced
  accuracy, and 0.08900 macro Brier.
- Retain both Muon `5e-5` and `1e-4` for replicated 50%-data confirmation. Keep
  AdamW `5e-5` as the control. Do not promote Muon or evaluate a selected recipe
  on competition validation from this single-seed, non-lineage-grouped screen.
- Treat the reversal across rates as evidence that RMS matching does not remove
  optimizer-specific learning-rate sensitivity: Muon was worse at `1e-5` and
  `2e-5`, and better in macro AUROC from `5e-5` through `2e-4`.
