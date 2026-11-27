# Screen structural training interventions

Date: 2026-08-14. Status: recorded in the decision log.

- After conventional optimizer and regularization changes failed to improve
  macro AUROC, test twenty rank-128 cells on the fixed internal development
  protocol: three baseline seeds and seventeen seed-0 interventions spanning base
  precision, adapter initialization/parameterization/targets, loss geometry,
  adaptive group weighting, and effective batch size.
- Train on all 11,177 rows outside the frozen 1,972-row development split. Keep
  competition validation and the final test untouched. Require a positive
  seed-matched AUROC delta and the existing 0.002 balanced-accuracy/Brier gates,
  then seeds 1 and 2, before promotion.
- Evaluate retained half-epoch checkpoints and predeclared checkpoint/seed logit
  ensembles. Cross-fit threshold and Platt diagnostics within development;
  treat them as operating-point/calibration evidence rather than ranking gains.
- Use causal Transformers evaluation across the screen because pinned vLLM does
  not support DoRA. Treat LoftQ as non-promotable diagnostic evidence on its NF4
  inference base, and require a separate serving-parity canary for any selected
  nonstandard adapter.
- Treat the two explicit decision-head cells as readout/gradient-estimator
  ablations: their forward scores and head gradients use the auxiliary head,
  while the QLoRA backbone receives an explicit straight-through gradient via
  the literal `0|1` LM-head rows. FLA 0.5.2 cannot lower direct auxiliary-head
  gradients into Qwen3.5's gated-delta blocks; record the surrogate in metadata
  and evaluate only the auxiliary head.
