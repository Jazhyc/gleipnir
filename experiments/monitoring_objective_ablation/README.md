# Monitoring objective ablation

Status: five-arm, one-seed ID-development screen frozen before training.

## Question

Can the selected monitoring-only Qwen3.5-4B Kimi-soft recipe gain complementary
evidence sensitivity from the paper's already-stored structured analyses or
from action-level multiple-instance learning (MIL)? This experiment makes no
new teacher calls and keeps the cached Kimi K3 probability as the primary
target in every arm.

## Interventions

Two arms add assistant-only cross-entropy on the paper-author Gemini 2.5 Pro
analysis selected by a Claude Sonnet 4.5 quality judge. The direct Kimi forward
continues to use Gleipnir's unchanged compact `Prediction:` prompt; the
rationale forward uses the original system prompt and trajectory stored in the
three-message source row. The rationale's terminal ordinal answer is known to
have been selected or regenerated to agree with the source label, so it is
privileged explanatory supervision rather than an independent calibrated
target.

Three arms add MIL over at most eight deterministic action/tool-result
endpoints. Each endpoint is an unlabeled instance; a length-normalized pool
produces the bag logit and is trained against the same Kimi soft probability as
the complete trajectory. Max, log-mean-exp, and top-three-mean pooling test
sparse, smooth, and repeated-evidence assumptions. MIL is weak localization,
not contrastive learning.

All arms use the selected `2e-5` learning rate, one epoch, seed 0, rank-128
QLoRA, effective batch 32, the proven selective-checkpointing/compilation
recipe, and pinned FLA/causal-conv1d kernels. Rationale inputs are capped at
27,648 tokens while the primary Kimi input remains at 29,696. This preserves
every rationale target (maximum 1,744 tokens) and full rationale context for
more than 95% of rows. The cap is needed because the original rationale backward
peaked at 77.5 GiB. Fully eager SDPA attempted a 52.2 GiB attention allocation,
and compiling an all-layer-checkpointed model attempted a 45.6 GiB quadratic
attention allocation even after the cap. Compiling a shell around the
full-attention graph break also made flash SDPA ineligible. All arms therefore
compile only the 24 linear-attention shells while holding FLA/causal-conv behind
graph breaks; the 8 full-attention layers remain eager. Unpadded batches omit
the redundant all-ones attention mask, and the campaign disables all SDPA
fallbacks. Trainer is explicitly prevented from globally re-enabling
checkpointing after the selective policy is applied. A 2,048-token
BF16-autocast eager/compiled logit canary must pass before training. The exact
arms and stop conditions are in `config.yaml`.

## Evaluation and promotion

Evaluate every arm on the frozen 3,012-row STRIDE-test plus
Gloom-Exfiltration ID suite. The primary metric is two-source macro normalized
pAUROC@20. An arm must improve at least 0.005 over the existing `2e-5` control,
avoid a source regression worse than 0.01, and avoid macro-Brier regression
worse than 0.005. Strict OOD remains untouched.

If at least one arm passes, run only the smallest motivated combination of the
best rationale weight and best MIL pool. A one-seed result remains exploratory
and requires replicated confirmation before promotion.

## Commands

On the prepared Lambda checkout:

```bash
python -m experiments.monitoring_objective_ablation.prepare
python -m experiments.monitoring_objective_ablation.run_lambda
```

Runtime status and artifacts are written below
`results/monitoring_objective_ablation/`.
