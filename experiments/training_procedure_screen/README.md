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

## Hard-anchor seed replication

The initial screen found that hard-label weights 0.10 and 0.25 improved seed-0
macro AUROC, while 0.25 gave the best AUROC and Brier and 0.10 gave the best
final balanced accuracy. Before using an untouched confirmation set, replicate
only these two pre-existing weights at seeds 1 and 2. Do not introduce another
anchor weight in this phase. Compare each result with its same-seed baseline,
then rank the three-seed means by macro AUROC subject to the original mean BA
and Brier regression limits of 0.002.

The effective-batch-64 cell is not replicated in this phase. Although it raised
seed-0 macro AUROC, it reduced balanced accuracy beyond the frozen acceptance
limit and is a weaker signal than either hard anchor. This keeps the replication
focused and limits further adaptation to the internal development split.

```bash
python experiments/training_procedure_screen/prepare_hard_anchor_replication.py
python experiments/training_procedure_screen/run_hard_anchor_replication_lambda.py
```

The four jobs reuse the exact rank-128 QLoRA recipe, full training set, and
internal development set from the screen. Competition validation and final test
remain prohibited. Passing this replication permits choosing an internal
candidate, not claiming held-out confirmation; the selected recipe must then be
frozen before a one-shot evaluation on an untouched split.

## Hard-label strength curve

Queue one fixed response-curve campaign behind the replication so intermediate
development scores cannot alter its design. Reuse the three-seed results at
direct-loss weights 0, 0.10, and 0.25. Add three seeds at weights 0.025, 0.05,
0.15, 0.20, 0.35, 0.50, 0.75, 1, 2, and 4, plus a hard-label-only endpoint
with the soft-teacher loss disabled. With soft-loss weight 1, these settings
span effective hard-label fractions from 2.4% through 80%; the final endpoint
is 100% hard-label training.

Choose among the resulting fourteen strengths using three-seed mean macro AUROC
subject to the original paired mean BA and Brier constraints. Do not extend the
grid after viewing results. This is still selection on the repeatedly inspected
internal development split, so the curve can select only a candidate for one
future evaluation on untouched data. Save and evaluate final adapters only;
checkpoint selection is deliberately excluded from this tuning phase.

```bash
python experiments/training_procedure_screen/prepare_hard_label_strength.py
python experiments/training_procedure_screen/run_hard_label_strength_lambda.py
```

The exact causal evaluator also supports an unadapted-base diagnostic on this
same development split. Run it in an environment where the pinned FLA kernels
have already been installed and added to `PYTHONPATH`:

```bash
python experiments/training_procedure_screen/evaluate_causal.py \
  --base-only-output results/training_procedure_screen/unadapted_base
```

This comparison measures adaptation gain on the internal split only. It is not
evidence of out-of-distribution transfer and must not be used to select another
training setting.

### Hard-label strength outcome

The complete three-seed curve finished on 2026-08-17. Under the predeclared
macro-AUROC-first rule, direct-loss weight `0.50` is the internal winner. With
soft-loss weight `1.0`, this is an effective target containing one-third hard
label and two-thirds Kimi probability. Relative to the paired soft-only seeds,
it improved mean macro AUROC by `0.01014`, balanced accuracy by `0.01517`, and
Brier by `0.01706` (lower is better). Its absolute three-seed means were
`0.97714` AUROC, `0.92645` balanced accuracy, and `0.05480` Brier.

The AUROC peak is broad rather than sharply identified. Weight `0.50` exceeds
the previously replicated `0.25` anchor by only `0.00054` mean AUROC and loses
to it on seed 1, although weight `0.50` improves balanced accuracy and Brier on
all three paired seeds. Weight `2.0` reaches `0.97660` AUROC while improving
balanced accuracy to `0.94099` and Brier to `0.04707`, so it is a secondary
operating-point candidate rather than the primary selection. It Pareto-dominates
hard-only training, which reaches `0.97256` AUROC, `0.93551` balanced accuracy,
and `0.05543` Brier. The combined results therefore support complementary soft
and hard supervision, but not a precisely estimated optimum.

A matched unadapted Qwen3.5-9B causal evaluation on the same 1,972 rows reached
`0.92273` macro AUROC, `0.85373` macro balanced accuracy, and `0.09790` macro
Brier. Its pooled AUROC was only `0.88903`; comparisons must therefore keep the
macro-versus-pooled aggregation fixed. Relative to this base, the three-seed
hard-only mean improves macro AUROC by `0.04983`, balanced accuracy by `0.08178`,
and Brier by `0.04246`.

The gain is not evenly distributed. Base-to-hard-only AUROC rises from `0.99356`
to `1.00000` on instructed Aletheia datasets, `0.88204` to `0.93589` on varied
Aletheia datasets, and `0.80178` to `0.97275` on Liars Bench. The very large
Liars gain and near-saturated instructed family make the aggregate hard-only
number compatible with source/template adaptation rather than broad transfer.
The mixed weight-0.50 candidate is stronger on the varied family (`0.95078`),
so retain soft supervision and test both recipes on a source-grouped untouched
surface before making a generalization claim.

Do not extend the internal grid. Freeze weight `0.50` as the AUROC-selected
candidate and treat weight `2.0` only as a predeclared-style threshold/calibration
trade-off for future work. Neither may be promoted without one evaluation on an
untouched, preferably source-grouped split; competition validation and the final
test were not used in this campaign.
