# Structural training-procedure screen

This experiment tests the remaining training-side levers after learning-rate,
Muon, warmup, scheduler, horizon, dropout, weight decay, target-temperature, and
fixed dataset-sampling screens failed to improve the selected AdamW recipe.
It emphasizes structural changes to the adapter, loss, and gradient estimator,
plus diagnostics that reuse checkpoints rather than spending more training.

## Hypothesis and frozen interventions

The hypothesis is that at least one structural intervention improves macro
AUROC over a seed-matched rank-128 baseline without regressing macro balanced
accuracy or Brier by more than 0.002. The twenty frozen training cells are:

- the current QLoRA recipe at seeds 0, 1, and 2;
- a seed-0 BF16-base LoRA control;
- LoftQ, EVA, and DoRA adapter variants;
- LoRA on all current projections plus `lm_head`, attention projections only,
  and MLP projections only;
- a trainable two-row decision projection initialized from the literal `0|1`
  LM-head rows and a separately initialized binary classification head;
- soft BCE with hard-label weights 0.10 and 0.25;
- teacher-margin Huber loss with deltas 0.5 and 1.0;
- EMA-smoothed GroupDRO weighting over dataset losses;
- effective batches 8, 16, and 64 around the baseline batch 32.

Every non-baseline cell is a single-factor seed-0 intervention. Model, rank 128
/ alpha 256, AdamW `5e-5`, linear schedule, 3% warmup, two epochs, direct
selected-position scoring, and examples remain fixed unless they are the named
intervention. Pairwise loss is intentionally excluded because the precursor
ablation overfit.

## Data and selection

Reuse the fixed 1,972-row dataset-label-stratified internal development split
created before the previous regularization screen. Train on its entire
11,177-row complement. The source rows expose no conversation or generator
lineage key, so lineage grouping remains unavailable. Competition validation
and the final test are prohibited.

Compare single-seed cells with baseline seed 0. A cell is eligible only if its
macro-AUROC delta is positive, macro balanced-accuracy regression is at most
0.002, and macro-Brier regression is at most 0.002. No cell can be promoted
until it is trained with seeds 1 and 2. LoftQ is a diagnostic only because its
quantization-error adapter must be evaluated on the NF4 base rather than the
project's ordinary BF16 serving base.

## Checkpoint, ensemble, and calibration diagnostics

Retain adapters about every half epoch and evaluate every checkpoint causally
against the same BF16 base, except the declared NF4 LoftQ diagnostic. Predeclare
logit averaging across the three final baseline seeds and across all or the last
two baseline-seed-0 checkpoints. These ensembles are evaluated without choosing
members from their scores.

For each baseline seed, cross-fit a global balanced-accuracy threshold and a
Platt calibration map over five deterministic identity folds. These diagnostics
can improve the operating point or Brier but are not AUROC interventions and do
not turn this internal split into confirmation evidence.

## Implementation and serving qualifications

The trainer supports standard, LoftQ, and EVA initialization; DoRA; configurable
LoRA targets; row-wise BCE/Huber losses; and EMA GroupDRO weights. It records the
resolved adapter configuration, quantization, GroupDRO state, batch, FLA kernel,
memory, throughput, checkpoints, and learning-rate history.

All screen metrics use the causal Transformers path so DoRA, `lm_head` LoRA,
and the two explicit decision heads can be compared without silently dropping
unsupported weights. DoRA and the explicit heads are not currently accepted by
the pinned vLLM LoRA loader. Any winner must pass a separate serving parity
canary or adopt a documented causal classification serving contract before
promotion.

The pinned FLA 0.5.2 backward kernel cannot lower a gradient injected directly
from an auxiliary hidden-state head. For the two decision-head cells, the
forward score and head gradient come exactly from the declared two-row head,
while the LoRA backbone receives a straight-through surrogate gradient through
the literal `0|1` LM-head rows. The head consumes a detached copy of the exact
selected hidden states seen by the LM head. Evaluation uses only the auxiliary
head, and trainer metadata records `lm_token_straight_through`; these cells are
therefore readout/surrogate-gradient ablations rather than ordinary end-to-end
classification heads.

## Preflight, execution, and stop condition

Before the campaign, train two steps and run an eight-row causal canary for
BF16 LoRA, LoftQ, DoRA, EVA, `lm_head` LoRA, both decision heads, and GroupDRO.
Fail closed if any
risk cell fails, FLA 0.5.2 is unavailable, the fixed split overlaps, the
20-cell contract changes, metadata/checkpoints are absent, or causal evaluation
is incomplete.

```bash
python experiments/training_procedure_screen/prepare.py
python experiments/training_procedure_screen/run_lambda.py
```

The launcher balances estimated load over both Lambda H100s, evaluates jobs and
checkpoints in two causal lanes, and writes final, checkpoint, contrast,
ensemble, and calibration summaries under the ignored
`results/training_procedure_screen/` tree.
