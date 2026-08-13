# Scheduled AdamW versus RMS-matched Muon

This experiment screens whether Muon improves rank-64 QLoRA soft-distillation
optimization when its update magnitude and learning-rate schedule are made
comparable to AdamW.

## Hypothesis and intervention

The hypothesis is that, at the same scheduled base learning rate, Moonshot-style
RMS-matched Muon reaches higher internal-development macro AUROC or the same
AUROC with lower training loss than AdamW. The intervention is optimizer
(`adamw` or `muon`) crossed with base learning rates `1e-5`, `2e-5`, `5e-5`,
`1e-4`, and `2e-4`.

Muon applies Newton-Schulz orthogonalization to every two-dimensional trainable
LoRA tensor. For a matrix of shape `[A, B]`, its polar update is multiplied by
`0.2 * sqrt(max(A, B))`, targeting update RMS 0.2. Both optimizers read the same
scheduler-managed parameter-group `lr`; Muon does not have an independent,
unscheduled learning rate. The screen holds fixed a linear schedule, 3% warmup,
two epochs, rank 64 / alpha 128, QLoRA, effective batch 32, selected-position
logits, and the pinned FLA 0.5.2 causal training path.

## Held-out selection and limitations

Hyperparameters are selected on a new internal development split, never on the
822-row competition validation set. Training uses the stable seed-0 25%
dataset-label-stratified selection (3,287 rows). Development is a deterministic
20% stratified selection from its complement, approximately 15% of the original
training pool. The two identity sets must be disjoint.

The source artifact exposes dataset and row identity but no conversation or
generator-lineage identifier. Consequently this is not a true lineage-grouped
holdout, and that limitation is recorded in the manifest. The screen is for
pruning only: its winner requires replicated 50%-data confirmation before any
single evaluation on competition validation.

## Metrics and stop condition

Primary selection uses macro per-dataset AUROC. Macro Brier score and balanced
accuracy at 0.5 are constraints; pooled metrics, training loss, throughput, and
peak GPU memory are retained. Report Muon-minus-AdamW contrasts at each shared
learning rate.

Run the ten predeclared cells once. Stop and mark the screen invalid if the
training/development identities overlap, either optimizer lacks a paired rate,
Muon does not use `match_rms_adamw`, the FLA kernel falls back, scheduling does
not mutate Muon's applied `lr`, or any internal-development score is missing.
Do not select from competition validation or add rates after seeing results.

## Prepare and launch

```bash
python experiments/optimizer_lr_scaling/prepare.py
python experiments/optimizer_lr_scaling/run_lambda.py
```

Ignored materialized inputs live in `data/optimizer_lr_scaling/`, model outputs
in `results/optimizer_lr_scaling/`, and logs in
`logs/lambda/optimizer_lr_scaling/`. The two Lambda lanes run matched AdamW and
Muon rates concurrently, beginning at the current `5e-5` baseline. Evaluation
uses one persistent vLLM process and writes `summary/results.csv`,
`summary/paired_contrasts.csv`, and `summary/summary.json`.
