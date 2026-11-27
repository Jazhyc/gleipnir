# Run tool-trajectory scaling as Kimi-soft-only distillation

Date: 2026-08-30. Status: recorded in the decision log.

- Train only Kimi K3 soft-logit students for the first Qwen3.5-9B
  tool-trajectory scaling campaign. Do not train matched hard-label adapters;
  retain source labels solely for provenance, stratification, and evaluation.
- Use one seed and rank-128 QLoRA at the six paper row counts, but train every
  point for exactly one epoch rather than copying the paper's optimizer-step
  horizons. This targets the effect of adding unique data without repeating the
  smallest subsets for 63 or 25 epochs. The paper rationale curve remains an
  external descriptive reference, not a supervision-channel causal comparison.
- Add one adapter over the full 8,688-row tool-trajectory pool plus all 13,149
  prior deception soft-distillation rows, also for one epoch. Treat the result
  as fixed-epoch data scaling rather than a fixed-compute mixture ablation.
- The prior single-seed deception optimization screen favored two epochs over
  one by 0.00470 macro AUROC but worsened Brier by 0.00144. This indirect,
  shorter-context evidence does not justify doubling the trajectory campaign;
  prioritize the controlled data-volume curve and timely completion.
- Preserve effective batch 32 while changing the long-context realization to
  microbatch 1 and accumulation 32, subject to a rank-128 longest-sequence H100
  preflight and a zero-truncation 29,696-token preparation audit.
