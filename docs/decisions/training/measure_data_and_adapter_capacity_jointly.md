# Measure data and adapter capacity jointly

Date: 2026-08-12. Status: recorded in the decision log.

- Cross nested `10%`, `25%`, and `50%` training selections with QLoRA ranks 4,
  16, 64, and 256 using paired seeds 0, 1, and 2. Reuse the matching 100%-data
  cells from the completed capacity sweep, yielding 36 new training jobs and 12
  checksum-preserved references.
- Hold the proven NF4/double-quantized/BF16, FLA 0.5.2, microbatch-8 recipe fixed
  so the only interventions are data volume and adapter rank. Keep two epochs,
  meaning optimizer updates scale with data volume rather than being held fixed.
- Treat macro AUROC and within-seed rank contrasts as primary evidence. Estimate
  whether the paired rank-256 advantage over rank 16 grows with log training
  rows. Do not select a deployment rank on this validation surface.
