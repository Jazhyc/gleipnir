# Joint data-volume and QLoRA-rank scaling

This experiment tests whether the adapter rank needed to saturate Qwen3.5-9B
monitor performance increases with annotated training-data volume. It crosses
three new nested Phoenix 8.1 fractions (`10%`, `25%`, and `50%`) with ranks `4`,
`16`, `64`, and `256`, using seeds `0`, `1`, and `2`. The twelve matching
`100%`-data results are reused from the completed QLoRA adapter-capacity sweep,
so only 36 new adapters are trained.

## Hypothesis and estimands

The preregistered qualitative hypothesis is that the paired AUROC advantage of
high rank over rank 16 increases with training rows. The primary estimands are:

- macro per-dataset AUROC at every data × rank cell;
- within-seed AUROC contrasts of ranks 4, 64, and 256 against rank 16;
- the slope of paired `rank 256 - rank 16` AUROC against log training rows;
- the observed best mean-AUROC rank and a bounded rank curve at each fraction.

A positive interaction slope supports increasing useful capacity with data. A
flat or negative slope rejects that hypothesis for this range and recipe. The
best observed rank is descriptive rather than a promotion choice because the
same validation set is used to estimate the surface.

## Intervention and controls

Selections use the data-scaling experiment's stable hash order within each
`dataset × hard-label` stratum. They are nested within a seed, and the same
selection is shared across ranks for paired comparisons. Rank uses
`lora_alpha = 2 × rank`. Every cell otherwise holds fixed:

- Qwen3.5-9B and Kimi K3 soft BCE targets;
- two epochs, AdamW, learning rate `5e-5`, and effective batch size 32;
- causal text-only training with selected-position logits;
- NF4 QLoRA with double quantization and BF16 compute;
- pinned FLA 0.5.2 gated-delta kernels and gradient checkpointing;
- the checksum-rebased multimodal serving adapter used by vLLM evaluation.

Holding epochs fixed makes this a data-and-capacity scaling experiment rather
than a fixed-optimizer-step experiment: larger fractions deliberately receive
more updates. The prior full-data reference used the identical recipe and is
admissible only if all ranks, seeds, input checksums, and validation identities
match.

## Held-out rule and stop condition

Evaluation uses the frozen 822-row, 21-dataset competition validation split.
It is not used for early stopping, threshold tuning, or subset selection. The
final competition test split remains untouched.

Run the 36 predeclared jobs once. Stop and mark the surface invalid if selections
are not nested, a selection checksum changes, train and validation identities
overlap, FLA falls back, any cell or direct score is missing, or the referenced
full-data results do not provide exactly the same three seeds and four ranks.
Report rejected power-law fits, non-monotonicity, calibration degradation, and
negative interactions rather than adding post-hoc fractions or ranks.

## Prepare and run on Lambda

```bash
python experiments/data_capacity_scaling/prepare.py
python experiments/data_capacity_scaling/run_lambda.py
```

The campaign writes ignored artifacts to
`results/data_capacity_scaling_qlora/`, runtime logs to
`logs/lambda/data_capacity_scaling_qlora/`, and atomic progress to
`results/data_capacity_scaling_qlora/lambda_status.json`. It fills both H100s
with longest-processing-time scheduling, evaluates all new adapters in one
persistent vLLM process, and produces:

- `summary/replicates.csv`;
- `summary/surface.csv`;
- `summary/paired_contrasts.csv`;
- `summary/interaction_summary.json`.

Training and evaluation are resumable. The full-data adapters remain in their
original artifact tree and are referenced rather than copied.
