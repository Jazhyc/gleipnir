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
