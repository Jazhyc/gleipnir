# Adapter-capacity scaling on full Kimi data

This experiment measures how Qwen3.5-9B monitor quality changes with trainable
capacity while holding the 13,149-row Phoenix 8.1 dataset and optimization recipe
fixed.

## Hypothesis and scaling form

For LoRA rank `r`, the residual validation error follows a bounded power law:

```text
M(r) = M_inf - A * r^(-alpha)
```

The fit uses ranks `1, 2, 4, 8, 16, 32, 64, 128, 256`. Full fine-tuning is an
endpoint comparison and is not assigned a fictitious rank or included in the
power-law fit. Rank and measured trainable parameter count are both recorded.

The primary metric is macro per-dataset AUROC. Macro balanced accuracy at the
fixed `0.5` threshold is co-primary diagnostic evidence. Pooled metrics,
calibration, score ties, per-dataset results, and the instructed/varied scenario
views are retained in each result.

## Intervention and controls

Each LoRA rank uses `lora_alpha = 2 * rank`, keeping `alpha/r = 2` fixed. Target
modules are `q`, `k`, `v`, `o`, gate, up, and down projections in the Qwen3.5
language model. All conditions use the same Qwen3.5-9B image-text-to-text loader,
full training data, Kimi K3 soft targets, two epochs, AdamW at `5e-5`, effective
global batch size 32, and seeds `0, 1, 2`.

The full-fine-tuning condition marks every conditional-generation checkpoint
parameter trainable and uses two-H100 FSDP2 full sharding. Since inputs are
text-only, the visual tower receives no task gradient; this endpoint tests full
backbone optimization for the observed text-monitoring workload, not learned
visual adaptation.

## Held-out rule and stop condition

Evaluation is frozen to the same 822-row, 21-dataset competition validation set
as the data-scaling experiment. It is not used for fitting, early stopping,
threshold tuning, or recipe selection. The final competition test split remains
untouched.

The original design predeclared 30 jobs: 27 LoRAs and three full fine-tunes. The
full-finetuning endpoints may be explicitly deferred until the LoRA curve shows
that more trainable capacity is useful. Stop and mark the run invalid if input
checksums change, train/validation identities overlap, any scheduled condition
fails, parameter counts are absent, or any scheduled score is missing. Reject
the LoRA curve as an extrapolator if its exponent is non-positive or its
asymptote lies outside `[0, 1]`. Report non-monotonic and negative results.

## Lambda schedule

Prepare the immutable jobs, then launch on the reserved two-H100 instance:

```bash
python experiments/adapter_capacity_scaling/prepare.py
python experiments/adapter_capacity_scaling/run_lambda.py
```

To finish and evaluate the LoRA curve while intentionally cancelling all full
fine-tuning jobs, use:

```bash
python experiments/adapter_capacity_scaling/run_lambda.py --skip-full
```

The scheduler first fills both GPUs with rank jobs using
longest-processing-time balancing. Unless `--skip-full` is set, it then runs the
three full fine-tunes across both GPUs. Training and evaluation are resumable
from completion markers, and progress is written atomically to
`results/adapter_capacity_scaling/lambda_status.json`.

Outputs are written under `results/adapter_capacity_scaling/`; runtime logs are
written under `logs/lambda/adapter_capacity_scaling/`. Neither artifact tree is
committed.
