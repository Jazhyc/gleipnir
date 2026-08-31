# Kimi-soft rank-128 distillation scaling

This campaign trains `Qwen/Qwen3.5-9B` action-only classifiers from the frozen
Kimi K3 literal `0|1` distributions. It is intentionally **soft-only**. There
are no hard-label student arms in this run; source labels remain in the cache
only for provenance, stratification, and later evaluation. Consequently, this
campaign estimates a Kimi-soft data-scaling curve and must not be described as
a matched hard-versus-soft supervision ablation.

## Frozen design

The six tool-trajectory points use seed 0, rank 128 (`alpha=256`), and the
paper's row counts: 204, 504, 996, 2,004, 4,008, and 8,688. Every point trains
for exactly one epoch. The campaign deliberately does not reproduce the
paper's optimizer-step schedule: its small points repeat the same rows up to 63
times, while the present estimand is the effect of adding unique data under a
fixed epoch count. Subsets are exact, nested, and approximately equal over the
ten source-label strata until the full point, which uses the paper author's
natural 8,688-row mixture.

The seventh adapter adds all 13,149 prior Aletheia deception-distillation rows
to all 8,688 tool-trajectory rows and also trains for one epoch. Total updates
therefore grow with unique data volume across the entire experiment. This is a
fixed-epoch data-scaling design, not a fixed-optimizer-step design.

The earlier single-seed deception optimization screen found that two epochs
improved macro AUROC by 0.00470 over one epoch, while worsening Brier score by
0.00144. That shorter-context result is not direct evidence for this benchmark;
one epoch is used here to prioritize a timely, controlled comparison of unique
data volume.

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

## Mixed-data Qwen3.5-4B transfer follow-up

After the seven Qwen3.5-9B jobs, one separately identified follow-up trains
`Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on the complete 21,837-row mixed
artifact. This remains Kimi-soft-only training; no hard-label loss or arm is
introduced. The intervention transfers the 9B recipe without a 4B
hyperparameter search: seed 0, one epoch, rank 128 (`alpha=256`), AdamW at
`5e-5`, linear decay with 3% warmup, zero weight decay/dropout, selected-token
soft BCE, NF4 double-quantized QLoRA with BF16 compute, microbatch 1 with
accumulation 32, and the 29,696-token cap.

The pinned 4B and 9B `tokenizer.json` and `tokenizer_config.json` artifacts are
byte-identical, so the existing zero-truncation token audit applies. A distinct
rank-128 longest-32 preflight still checks the 4B model loader, pinned FLA 0.5.2
kernels, memory, and adapter export before the full job. The hypothesis is that
the mixed-data intervention transfers usefully to the smaller backbone. Its
primary baselines are the matched 9B mixed adapter and the frozen unadapted 4B
evaluation; this launch is not evidence that the inherited recipe is optimal
for 4B.

Prepare and run this follow-up with:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_mixed_4b_distillation
python -m experiments.tool_trajectory_monitoring.run_mixed_4b_lambda
```

Artifacts and status live under
`results/tool_trajectory_distillation_mixed_qwen4b/`; Lambda logs live under
`logs/lambda/tool_trajectory_distillation_mixed_qwen4b/`.
Held-out STRIDE test and Gloom-Exfiltration will be used for later checkpoint
selection; all six strict OOD sources remain untouched during training.

## Mixed-data Qwen3.5-0.8B lower-bound follow-up (stopped)

This separately identified follow-up was designed to test whether the benefit
of Kimi-soft distillation grows further as student capacity falls. It targeted
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17` on the same complete 21,837-row
mixed artifact, followed by evaluation of both the unadapted base and final
adapter on all 6,395 frozen strict-OOD rows. The intervention was soft-only: no
hard-label, completion, or rationale loss was enabled.

The statistical recipe transfers unchanged: seed 0, one epoch, rank 128
(`alpha=256`), AdamW at `5e-5`, linear decay with 3% warmup, zero weight
decay/dropout, selected-token soft BCE, NF4 double-quantized QLoRA with BF16
compute, effective batch 32, and maximum length 29,696. The smaller model uses
an H100 systems batch of microbatch 1 with accumulation 32, subject to a
rank-128 longest-32 preflight. This changes only how the effective batch is
packed, not the optimizer exposure.

A matched longest-32 systems benchmark rejected microbatch 8 because padding
mixed-length examples together made one optimizer step take 189.54 seconds and
consume about 80 GB. QLoRA microbatch 1 with accumulation 32 took 51.44 seconds
after kernel warm-up and about 10 GB on the identical rows. The two QLoRA
layouts exported byte-identical FP32 master and serving adapter hashes. A
matched BF16 LoRA attempt took 75.78 seconds, so removing quantization was also
rejected.

The full QLoRA run stabilized around 15--18 seconds per optimizer step,
projecting roughly 2.8--3.3 hours for 683 steps, versus 5.49 hours for the
completed 4B campaign. The run was stopped by user decision at step 12 of the
final restart: the approximately 1.7--2.0x systems speedup was not sufficient
to make the additional frontier point decision-relevant after Gleipnir 4B had
already established the low-cost frontier. No full checkpoint was saved, the
OOD handoff never started, and no 0.8B performance claim should be made.

The 0.8B tokenizer vocabulary is byte-identical to the pinned 4B/9B
vocabulary. Its revised chat template defaults to non-thinking but renders the
same explicit `enable_thinking=false` assistant boundary. The primary metric is
source-balanced Mean-OOD pAUROC@20; regular AUROC, pooled metrics, per-source
metrics, calibration, ties, and threshold diagnostics remain required. Neither
OOD labels nor the unadapted result may select a checkpoint or alter the prompt.

Stop on input or revision drift, any non-soft objective, FLA fallback, preflight
OOM, incomplete FP32-master or serving exports, parity failure, prompt
truncation, missing decision-token logprobs, incomplete OOD coverage, or a
non-finite score. The inherited hyperparameters are an intervention, not a claim
of 0.8B optimality.

The historical preparation and launch commands were:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_mixed_08b_distillation
python -m experiments.tool_trajectory_monitoring.run_mixed_08b_lambda
python -m experiments.tool_trajectory_monitoring.run_distilled_ood_lambda \
  --config experiments/tool_trajectory_monitoring/distillation_mixed_qwen08b_ood_benchmark.json \
  --result-dir results/tool_trajectory_distillation_mixed_qwen08b_ood \
  --status results/tool_trajectory_distillation_mixed_qwen08b_ood/status.json
```

Training artifacts live under
`results/tool_trajectory_distillation_mixed_qwen08b/`; OOD artifacts live under
`results/tool_trajectory_distillation_mixed_qwen08b_ood/`. The authoritative
runtime disposition is `stop_record.json`; do not infer an experiment failure
from the launcher's expected SIGTERM traceback.
