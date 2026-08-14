# Optimization and regularization screen

This experiment screens sixteen rank-128 AdamW QLoRA recipes after the
full-data Muon confirmation found no robust optimizer advantage. It tests
training trajectory, LoRA regularization, teacher-target smoothing, and
macro-aligned dataset sampling while keeping the model, objective, batch, and
training examples fixed.

## Hypothesis and design

The hypothesis is that at least one of these levers improves internal-development
macro AUROC without regressing macro balanced accuracy or Brier by more than
0.002 relative to the shared baseline:

- warmup ratios `0`, `0.03`, and `0.10`;
- linear and cosine decay at `5e-5`, plus constant-with-warmup at `2.5e-5` to
  approximately match the linear schedule's integrated learning rate;
- one, two, and three training epochs;
- LoRA dropout `0`, `0.025`, `0.05`, and `0.10`;
- weight decay `0`, `0.01`, and `0.10`;
- proportional, square-root-balanced, and uniform-dataset sampling;
- teacher-logit scales `1`, `1.5`, and `2`.

The frozen design is one baseline plus fifteen single-factor interventions, not
a Cartesian product. All cells use seed 0, rank 128 / alpha 256, AdamW, the
standard QLoRA/FLA selected-logit path, and effective batch 32. The constant
schedule's half-sized learning rate avoids conflating schedule shape with an
approximately doubled integrated update.

## Held-out rule and limitations

A deterministic 15% dataset-label-stratified development split is removed from
the 13,149-row training artifact. Every screen cell trains on the same 50%
selection from the remaining tuning pool. Competition validation and the final
test are not used. The source rows lack conversation or generator-lineage keys,
so the split cannot enforce lineage grouping.

The screen is single-seed pruning evidence only. Rank eligible cells by macro
AUROC after applying the 0.002 balanced-accuracy and Brier constraints, retain
at most five non-baseline cells, and add seeds 1 and 2 before combining
interventions. Do not add cells after seeing screen results.

## Trainer instrumentation

Non-proportional sampling uses deterministic weighted sampling with replacement.
Square-root balancing gives each dataset total probability proportional to the
square root of its row count; uniform-dataset sampling gives every dataset equal
expected probability. Sampling mode and expected dataset mass are recorded.

Each job retains model-only causal adapter checkpoints approximately every half
epoch, capped at six, and records loss/learning-rate history, global steps,
checkpoint names, scheduler, warmup, dropout, target scale, sampling mode,
memory, and throughput. The final causal adapter is checksum-rebased for vLLM
serving as in the earlier scaling campaigns.

## Stop condition and execution

Stop if the training/development identities overlap, the 16-cell design changes,
FLA is unavailable, rank-128 batch 8 fails its preflight, a job loses its
recorded checkpoint/optimization metadata, or any development result is absent.

```bash
python experiments/optimization_regularization_screen/prepare.py
python experiments/optimization_regularization_screen/run_lambda.py
```

The launcher balances epoch-row load across both Lambda H100s, evaluates final
adapters in one persistent vLLM process, and writes `summary/results.csv`,
`summary/contrasts.csv`, and `summary/summary.json` under the ignored result
tree.

## Results

All sixteen cells completed on 2026-08-14. The campaign ran for 6 hours 16
minutes across the two H100 lanes, and neither competition validation nor the
final test was used. The shared baseline remained best on the predeclared
primary metric, with 0.96449 macro AUROC, 0.89841 macro balanced accuracy, and
0.07478 macro Brier. Every single-factor intervention reduced macro AUROC, so
this screen does not justify replacing or combining parts of the baseline.

The frozen eligibility and ranking rule produced this replication shortlist:

- square-root-balanced sampling: 0.96424 AUROC (-0.00025), 0.90416 balanced
  accuracy (+0.00575), and 0.07627 Brier (+0.00149);
- teacher-logit scale 1.5: 0.96359 AUROC (-0.00091), 0.90841 balanced accuracy
  (+0.01000), and 0.07336 Brier (-0.00142);
- weight decay 0.10: 0.96341 AUROC (-0.00108), 0.90511 balanced accuracy
  (+0.00670), and 0.07434 Brier (-0.00044);
- cosine scheduling: 0.96322 AUROC (-0.00128), 0.90100 balanced accuracy
  (+0.00259), and 0.07363 Brier (-0.00115);
- teacher-logit scale 2.0: 0.96252 AUROC (-0.00198), 0.90491 balanced accuracy
  (+0.00649), and 0.07366 Brier (-0.00112).

These are trade-offs, not promotions, and all are based on seed 0. In
particular, logit scale 1.5 is the strongest joint balanced-accuracy/calibration
alternative but gives up AUROC and increases tied scores; scale 2.0 is dominated
by scale 1.5 on all three macro metrics. LoRA dropout, the matched constant
schedule, and changing the training horizon did not produce a constraint-passing
primary-metric improvement. If this line is continued, add seeds 1 and 2 to the
frozen shortlist before testing combinations; otherwise retain the baseline
AdamW `5e-5`, linear schedule, 3% warmup, two epochs, zero dropout/weight decay,
proportional sampling, and unscaled teacher targets.
