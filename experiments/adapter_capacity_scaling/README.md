# QLoRA adapter-capacity scaling on full Kimi data

This experiment measures how Qwen3.5-9B monitor quality changes with trainable
capacity while holding the 13,149-row Phoenix 8.1 dataset and optimization recipe
fixed.

## Hypothesis and scaling form

For QLoRA rank `r`, the residual validation error follows a bounded power law:

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

Each QLoRA rank uses `lora_alpha = 2 * rank`, keeping `alpha/r = 2` fixed. Target
modules are `q`, `k`, `v`, `o`, gate, up, and down projections in the Qwen3.5
language model. The frozen base uses the standard QLoRA quantization recipe:
4-bit NF4 weights, double quantization, and BF16 compute. All rank conditions
use the Qwen3.5-9B causal text loader, full training data, Kimi K3 soft targets,
two epochs, AdamW at `5e-5`, microbatch 8, accumulation 4, effective batch size
32, and seeds `0, 1, 2`. Quantization is held fixed across the curve, so this
experiment estimates adapter-rank scaling under QLoRA rather than conflating
ordinary BF16 LoRA and QLoRA points.

Three paired seeds remain necessary. In the completed data-scaling experiment,
the 100%-data macro AUROCs were `0.96048`, `0.96167`, and `0.96405` (sample SD
`0.00182`, range `0.00357`); macro balanced accuracy had SD `0.00137`. The mean
macro-AUROC gain from 64% to 100% data was only `0.00256`, smaller than the
full-data seed range, and seed 2 averaged about `0.0014` higher across fractions.
Since plausible rank effects are of the same order, a single-seed curve could
mistake optimizer/init/shuffle noise for capacity scaling. Use the same three
seeds at every rank so rank contrasts can also be paired by seed.

Each FP32 causal adapter is preserved as the training master. At save time, a
checksum-tracked inference copy is produced by inserting `language_model` into
the PEFT key namespace. Tensor values are unchanged. Evaluation loads only the
rebased copy through vLLM's multimodal Qwen3.5 serving architecture on its
ordinary BF16 base; the base quantization is a training-memory intervention,
not part of the deployed adapter package.

## Throughput preflight

The inherited fast H100 recipe uses pinned `flash-linear-attention==0.5.2` and
`fla-core==0.5.2` gated-delta kernels, microbatch 8, accumulation 4, gradient
checkpointing, and `torch.compile=false`. It measured 8.811 samples/s for the
earlier rank-16 soft-distillation workload. The best eager Torch fallback
managed 4.279 samples/s, while the accidental microbatch-2 fallback used by the
superseded sweep managed only about 1.56 samples/s. The launcher installs FLA
without dependencies into `/tmp/gleipnir-qwen35-fla`, probes it before any model
import, and fails closed rather than silently using the Torch fallback.

Before scheduling the curve, rank 256 must complete a matched 20-step QLoRA
preflight at microbatch 8 without OOM, confirm NF4/double-quantized/BF16 metadata,
and show that the concrete gated-delta implementation is under `fla.ops`.
Preserve the benchmark log and do not lower the high-rank batch silently; a
recipe change would alter the controlled intervention and requires a new
campaign root.

On two H100s, the earlier microbatch-2 matched rank-16 runs measured 300.7
seconds with ordinary full-sequence logits and 309.3 seconds when projecting
only the distinct final positions. The selected-position path was 2.9% slower
and its accumulated loss differed slightly (`2.397` versus `2.364`), reflecting
a different GEMM shape and numerical trajectory. However, the rank-256 QLoRA
batch-8 preflight reached about 69 GB and then OOMed when Accelerate attempted a
roughly 25 GB full-logit conversion. The production curve therefore uses
selected-position logits consistently at every rank. This is a memory-enabling
intervention for the proven batch, and it supersedes the earlier batch-2 choice
without hiding that negative timing result.

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

Prepare the immutable QLoRA jobs, then launch on the reserved two-H100
instance:

```bash
python experiments/adapter_capacity_scaling/prepare.py
python experiments/adapter_capacity_scaling/run_lambda.py
```

To finish and evaluate the LoRA curve while intentionally cancelling all full
fine-tuning jobs, use:

```bash
python experiments/adapter_capacity_scaling/run_lambda.py --skip-full
```

The fresh campaign writes under `results/adapter_capacity_scaling_qlora/`. The
cancelled native-multimodal attempt remains preserved under
`results/adapter_capacity_scaling/`, and the superseded causal BF16 attempt is
preserved under `results/adapter_capacity_scaling_causal/`. The scheduler first
probes the pinned FLA environment, then fills both GPUs with rank jobs using
longest-processing-time balancing. Unless `--skip-full` is set, it then runs the
three full fine-tunes across both GPUs. Training and evaluation are resumable
from completion markers, and progress is written atomically to
`results/adapter_capacity_scaling_qlora/lambda_status.json`.

Runtime logs are written under
`logs/lambda/adapter_capacity_scaling_qlora/`. Artifact trees are not committed.
