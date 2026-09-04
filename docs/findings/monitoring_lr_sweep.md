# Monitoring-only Qwen3.5-4B learning-rate screen

Status: seed-0 pruning screen completed and audited 2026-09-04.

## Finding

The learning rate transferred from the earlier mixed-data recipe is not optimal
for one-epoch, rank-128 QLoRA distillation on only the 8,688 monitoring
trajectories. Under the frozen ID-development selection rule, AdamW at `2e-5`
replaces the freshly trained `5e-5` control.

| Learning rate | Macro pAUROC@20 | Delta | Macro AUROC | Macro Brier | Eligible |
| ---: | ---: | ---: | ---: | ---: | :---: |
| `1e-5` | 0.865921 | +0.018776 | 0.958562 | 0.083537 | No |
| **`2e-5`** | **0.871066** | **+0.023922** | 0.957643 | **0.079273** | **Yes** |
| `5e-5` | 0.847144 | — | 0.951173 | 0.095650 | Control |
| `1e-4` | 0.858995 | +0.011851 | 0.953651 | 0.090651 | Yes |
| `2e-4` | 0.798329 | -0.048815 | 0.937820 | 0.088241 | No |

The selected rate improves Gloom pAUROC@20 from `0.731865` to `0.783833` and
reduces STRIDE pAUROC@20 from `0.962424` to `0.958300`. That `0.004124`
source regression is inside the predeclared `0.01` limit. Macro balanced
accuracy rises from `0.878401` to `0.900203`; recall rises from `0.788214` to
`0.847277`, while FPR rises from `0.031413` to `0.046872` at the fixed
threshold. Score ties fall from 1,434 to 1,333.

The monitoring-only `2e-5` cell is essentially tied with the previously
evaluated mixed-data Gleipnir 4B on ID ranking (`0.871066` versus `0.869194`
macro pAUROC@20), but its Brier score is worse (`0.079273` versus `0.074303`).
This cross-run comparison is descriptive because both the data mixture and
training recipe differ.

## Cross-data comparison at the transferred rate

The freshly retrained monitoring-only `5e-5` control can also be compared with
the released mixed-data Gleipnir 4B on the identical 3,012 rows and prompt. The
released model contains the same 8,688 monitoring examples plus 13,149
deception examples.

| Metric | Monitoring-only `5e-5` | Released mixed `5e-5` | Mixed minus monitoring-only |
| --- | ---: | ---: | ---: |
| Macro pAUROC@20 | 0.847144 | 0.869194 | +0.022049 |
| Macro AUROC | 0.951173 | 0.957679 | +0.006506 |
| Macro Brier | 0.095650 | 0.074303 | -0.021347 |
| Balanced accuracy | 0.878401 | 0.905482 | +0.027081 |
| Recall | 0.788214 | 0.850790 | +0.062576 |
| FPR | 0.031413 | 0.039826 | +0.008413 |

The ranking improvement occurs on both sources: Gloom pAUROC@20 rises by
`0.040470`, and STRIDE rises by `0.003628`. At the fixed threshold, the mixed
model corrects 151 control errors while introducing 56 new errors across 207
prediction flips, a net gain of 95 rows. Scores remain closely related
(`r=0.9640`) but are on average `0.0380` higher in the mixed model. The upward
shift is larger for positives than negatives on both sources, explaining the
large recall gain alongside a smaller FPR increase.

This is useful evidence that deception supervision transfers positively to
monitoring ID performance at the transferred rate. It is not a clean data-only
causal estimate. Both runs use the same backbone revision, seed, rank-128 QLoRA,
AdamW `5e-5`, linear schedule, 3% warmup, effective batch 32, one epoch, and
selected-position Kimi-soft loss. However, adding deception expands the epoch
from 272 to 683 optimizer steps and changes where monitoring examples fall on
the schedule. The new control also uses causal-conv1d, selective checkpointing,
and decoder-shell compilation, whereas the released run predates those systems
changes. Those interventions are intended to preserve the objective but can
alter the numerical training trajectory.

The fact that tuned monitoring-only `2e-5` recovers the released model's ID
ranking, while the released model retains better Brier and fixed-threshold
metrics, suggests two working hypotheses: added deception data provides useful
regularization or positive transfer at `5e-5`, and some of its aggregate ranking
benefit can instead be recovered by lowering the monitoring-only learning rate.
Separating transfer from extra updates would require a matched-update mixture
ablation rather than this cross-run comparison.

## Scope and next decision

This is a one-seed pruning result on two in-distribution source families. It
does not establish seed stability or OOD generalization, and the strict
six-source OOD suite was not consulted. Advance `2e-5` to replicated
confirmation against `5e-5`; do not promote it as the general monitoring
learning rate until that confirmation passes.

All five jobs completed 272 optimizer steps with the required FLA 0.5.2,
causal-conv1d 1.6.2.post1, Triton 3.7.1, rank-128 NF4 QLoRA, selective
checkpointing, and decoder-shell compilation. Each cell ran at 0.929–0.938
examples/s, took 2.57–2.60 hours, and peaked at 24.86 GiB allocated. The full
two-GPU campaign took 9.19 hours including the longest-context preflight,
serving parity, and sequential evaluation.

Artifact SHA-256 values are:

- summary: `dee8e862b0f50c4f64b3ce1e23b8c5af20e124916f90f72229db99e7bafecb0e`;
- result table: `6ab439b3431b07caa8f173b686aed0d5c59be464e1a85c6a5e909d0f48c3a273`;
- parity report: `2e8e0e17d8087926328ce424f3da07f312f31136e77ce902fd0093ed7c2e8d50`;
- preparation manifest: `a133f8ecaa9ad6bff8e95c8cc43417a170f0da7e7e0776e5ccfbec3177ca2191`.
