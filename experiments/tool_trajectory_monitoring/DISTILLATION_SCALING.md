# Kimi-soft rank-128 distillation scaling

This campaign trains `Qwen/Qwen3.5-9B` action-only classifiers from the frozen
Kimi K3 literal `0|1` distributions. It is intentionally **soft-only**. There
are no hard-label student arms in this run; source labels remain in the cache
only for provenance, stratification, and later evaluation. Consequently, this
campaign estimates a Kimi-soft data-scaling curve and must not be described as
a matched hard-versus-soft supervision ablation.

## Frozen design

The six tool-trajectory points use seed 0, rank 128 (`alpha=256`), and exact
paper row/optimizer-step pairs: `204/378`, `504/375`, `996/372`,
`2,004/372`, `4,008/750`, and `8,688/813`. Subsets are exact, nested, and
approximately equal over the ten source-label strata until the full point,
which uses the paper author's natural 8,688-row mixture.

The seventh adapter adds all 13,149 prior Aletheia deception-distillation rows
to all 8,688 tool-trajectory rows. It trains for three epochs, matching the full
tool-only point's epoch count. This deliberately increases both domain breadth
and total optimization compute, so it is an exploratory transfer condition,
not a fixed-compute causal estimate of adding unrelated data.

Every adapter uses the retained best soft-distillation recipe: AdamW at
`5e-5`, linear decay with 3% warmup, zero weight decay/dropout, soft BCE at the
literal decision boundary, selected-position projection, standard NF4 QLoRA
with double quantization and BF16 compute, and pinned FLA 0.5.2 gated-delta
kernels. The effective batch remains 32. Long trajectories require a
microbatch of 1 with accumulation 32 and a checksumed 29,696-token cap; the
preparation audit requires zero student-context truncation.

Before the curve, the launcher trains one optimizer step over the 32 longest
contexts at rank 128. An OOM, FLA fallback, checksum drift, non-nested subset,
non-soft target, changed effective batch, or missing final causal master
invalidates the launch. Runtime is resumable and every FP32 causal master is
checksum-rebased into a serving-layout copy, without downcasting.

Prepare locally:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_distillation_scaling
```

On the already-reserved `gleipnir-control` target, run:

```bash
python -m experiments.tool_trajectory_monitoring.run_distillation_lambda
```

Artifacts live under `results/tool_trajectory_distillation_scaling/`, data
materializations under `data/tool_trajectory_monitoring/distillation_scaling/`,
and runtime logs under `logs/lambda/tool_trajectory_distillation_scaling/`.
Held-out STRIDE test and Gloom-Exfiltration will be used for later checkpoint
selection; all six strict OOD sources remain untouched during training.
