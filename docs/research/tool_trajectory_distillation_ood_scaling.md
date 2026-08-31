# Soft-distillation OOD data scaling

Status: completed and audited 2026-08-31; one seed per point.

This experiment measures strict-OOD monitoring as matched tool-trajectory
training volume increases under the inherited deception recipe. The six
Qwen3.5-9B points use `204`, `504`, `996`, `2,004`, `4,008`, and `8,688`
nested, source-label-balanced rows. Every point is a rank-128 QLoRA trained for
exactly one epoch at learning rate `5e-5`, effective batch size 32, and maximum
sequence length 29,696. Supervision is only Kimi K3's soft literal-token
distribution: soft-loss weight is 1, direct/hard-label loss weight is 0, and no
completion loss is used.

Kimi targets were generated with the richer teacher prompt. Students were
trained on the compact student prompt and are evaluated with that same compact
student prompt. Thus the student-side train/evaluation interface is matched;
the privileged teacher prompt is not shown to the deployed student.

## Results

The matched Qwen3.5-9B curve is available as a publication-ready
[`SVG`](../../figures/tool_trajectory_ood_data_scaling.svg) with a
[`PNG`](../../figures/tool_trajectory_ood_data_scaling.png) fallback. Regenerate
both artifacts directly from the table below with
`python scripts/plot_tool_trajectory_ood_scaling.py`.

The primary metric remains the paper-compatible, unweighted six-source mean of
raw normalized pAUROC through FPR 0.20. Because that operating region is
strict, regular unweighted Mean-OOD AUROC is reported beside it and is equally
important for interpreting the data-scaling shape. Pooled metrics weight rows
rather than sources and also include cross-source ranking pairs, so they answer
a different question.

| Backbone | Training data | Rows | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Pooled pAUROC@20 | Pooled AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | matched | 204 | 0.698752 | 0.885792 | 0.506548 | 0.865576 |
| Qwen3.5-9B | matched | 504 | 0.707647 | 0.884558 | 0.620362 | 0.854309 |
| Qwen3.5-9B | matched | 996 | 0.811115 | **0.935323** | 0.745490 | 0.917952 |
| Qwen3.5-9B | matched | 2,004 | 0.790651 | 0.918147 | 0.694083 | 0.903821 |
| Qwen3.5-9B | matched | 4,008 | **0.836543** | 0.925396 | **0.786020** | **0.934073** |
| Qwen3.5-9B | matched | 8,688 | 0.822682 | 0.921507 | 0.724366 | 0.917421 |
| Qwen3.5-9B | matched + deception | 21,837 | 0.821183 | 0.921536 | 0.783663 | 0.935507 |
| Qwen3.5-4B | matched + deception | 21,837 | 0.782350 | 0.908584 | 0.701531 | 0.913513 |

Bold values mark the best matched-data 9B point in each metric column, not a
selection rule. Checkpoints and prompts were frozen before OOD scoring.

The matched-data curve is not monotonic under one seed. The 204- and 504-row
points are essentially flat in regular AUROC, followed by a large gain at 996
rows. Mean-OOD AUROC peaks at 996 rows (`0.935323`), whereas Mean-OOD pAUROC@20
peaks at 4,008 rows (`0.836543`). The 8,688-row point is below both peaks. This
supports a real data-sufficiency transition by roughly 1,000 matched examples,
but not a smooth scaling law or the claim that every additional example helps.

### Comparison with the paper's Qwen3.5 scaling curves

Sinha et al. report Qwen3.5-4B and Qwen3.5-27B rationale-SFT curves at the same
six nominal row counts and on the same six-source Mean-OOD pAUROC@20 metric.
The underlying values are not present in the paper's tables or its pinned Hugging
Face artifact. The values below were therefore reconstructed from the vector
paths in arXiv v1 Figure 7 and rounded to three decimals; they are approximate
reference values, not author-released tabular results.

| Backbone | Rows | Mean-OOD pAUROC@20 (digitized) |
| --- | ---: | ---: |
| Qwen3.5-4B | 204 | 0.628 |
| Qwen3.5-4B | 504 | 0.675 |
| Qwen3.5-4B | 996 | 0.666 |
| Qwen3.5-4B | 2,004 | 0.708 |
| Qwen3.5-4B | 4,008 | 0.697 |
| Qwen3.5-4B | 8,688 | 0.700 |
| Qwen3.5-27B | 204 | 0.766 |
| Qwen3.5-27B | 504 | 0.783 |
| Qwen3.5-27B | 996 | 0.779 |
| Qwen3.5-27B | 2,004 | 0.772 |
| Qwen3.5-27B | 4,008 | 0.785 |
| Qwen3.5-27B | 8,688 | 0.807 |

The source is Sinha et al., *Training Deliberative Monitors for Black-Box
Scheming Detection*, [arXiv:2605.29601v1, Figure
7](https://arxiv.org/html/2605.29601v1#S4.SS5). In the v1 source bundle the
embedded file is
`figures_new/sft_data_scaling_pauroc_val_ood_all_models_largefont.pdf`, with
SHA-256
`e0c3009b3df8b32b4ede0e4619af9b9eeb26e1488099d43bf411ad39a32273df`.
The digitization maps the vector path coordinates to the labelled OOD-axis tick
coordinates before rounding, so it does not depend on raster resolution.

This is an intentionally non-causal comparison. The paper trains generated
rationales with SFT, varies training from 63 epochs at 204 rows to 3 epochs at
8,688 rows, and selects checkpoints on validation. Gleipnir trains Kimi soft
decision distributions for exactly one epoch and evaluates the final adapter.
The subset construction and output interface also differ. What is comparable
is the nominal unique-row schedule, frozen OOD suite, and aggregate metric—not
the training compute or intervention. Descriptively, Gleipnir 9B is above the
paper's 4B curve at every row count and crosses the digitized 27B curve between
504 and 996 rows, remaining slightly above it thereafter. This suggests strong
sample efficiency but does not isolate whether the difference comes from soft
targets, backbone size, training recipe, checkpoint selection, or interface.

### Low-FPR behavior by OOD source

| Matched rows | AgentDojo | CUA-SHADE | IaC | MLE-Sabotage | Rogue Eval | SHADE-Arena |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 204 | 0.516801 | 0.755445 | 0.706504 | 0.990816 | 0.795329 | 0.427615 |
| 504 | 0.655909 | 0.756662 | 0.689114 | 0.984694 | 0.743900 | 0.415606 |
| 996 | 0.862071 | 0.788212 | 0.814807 | 0.989796 | 0.930703 | 0.481098 |
| 2,004 | 0.868920 | 0.778122 | 0.822024 | 0.996939 | 0.888420 | 0.389483 |
| 4,008 | 0.965453 | 0.782817 | 0.914281 | 0.996939 | 0.962368 | 0.397399 |
| 8,688 | 0.894969 | 0.792510 | 0.930125 | 0.998980 | 0.912652 | 0.406859 |

The main rise comes from AgentDojo, IaC, and Rogue Eval. MLE-Sabotage is near
ceiling even at 204 rows, CUA-SHADE moves only modestly, and SHADE-Arena remains
the limiting source. The 996-to-2,004 dip is driven chiefly by SHADE-Arena and
Rogue Eval; the 4,008-to-8,688 dip is chiefly AgentDojo and Rogue Eval. The
macro non-monotonicity is therefore not just metric noise from one pooled class
balance.

## Does adding deception data help?

The 21,837-row mixed condition adds 13,149 earlier deception rows to the same
8,688 matched rows. Relative to the matched-only 8,688-row 9B adapter, its
Mean-OOD pAUROC changes by `-0.001499` and Mean-OOD AUROC by `+0.000029`:
effectively zero at the source-balanced macro level. Pooled pAUROC improves by
`+0.059298` and pooled AUROC by `+0.018086`, showing a useful change in the
example-weighted/cross-source ranking without a general per-source gain.

| OOD source | Mixed-minus-matched pAUROC@20 | Mixed-minus-matched AUROC |
| --- | ---: | ---: |
| AgentDojo | -0.001759 | +0.000147 |
| CUA-SHADE | +0.010336 | -0.005241 |
| IaC | +0.002873 | -0.000965 |
| MLE-Sabotage | -0.003061 | -0.000612 |
| Rogue Eval | +0.055733 | +0.020248 |
| SHADE-Arena | -0.073116 | -0.013404 |

The Rogue Eval gain is offset by SHADE-Arena interference. The result does not
support adding this deception corpus as a general way to improve tool-monitor
OOD performance under the current recipe, although its pooled-ranking benefit
may matter for deployments whose traffic resembles the observed row mix.

The 4B mixed adapter trails the 9B mixed adapter by `0.038834` Mean-OOD
pAUROC@20 and `0.012952` Mean-OOD AUROC, while its hosted inference proxy is
about one third as expensive (`$0.2598` versus `$0.8659` per 1,000 examples).
It is therefore a strong lower-cost point rather than a quality-matched
replacement.

## Interpretation limits

Each point has one seed, so local reversals cannot be separated from seed
variance. One epoch also means data volume and optimizer updates co-vary: the
matched runs use 7, 16, 32, 63, 126, and 272 optimizer steps, while mixed uses
683. This is the intended equal-exposure comparison, but it cannot identify
whether a gain comes from diversity or simply more updates. Notably, mixed
training receives far more updates than matched-only 8,688 and still produces
no macro gain.

These results argue for retaining the 996- and 4,008-row recipes as efficient
candidates and adding seeds before treating the 8,688-row checkpoint as the
default. A future experiment that isolates data diversity should match update
counts as a separate intervention rather than retroactively changing this
one-epoch campaign.

## Evaluation and audit

All adapters were scored on all 6,395 rows of the paper's six-source strict OOD
suite with native thinking disabled and normalized next-token logits for
literal `0` versus `1` after `Prediction:`. Mean prompt length was 8,654.7113
tokens (maximum 29,570); no prompt reached the 32,768-token limit.

Master-checkpoint versus rebased-serving parity passed before each backbone:

| Backbone | Adapter Pearson | Mean absolute score difference | Maximum vLLM adapter effect |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B | 0.999521 | 0.002957 | 0.328244 |
| Qwen3.5-4B | 0.998322 | 0.008621 | 0.701442 |

The final local audit verified identical unique ID coverage across all eight
prediction files, finite scores, prompt and adapter hashes, soft-only training
weights, and passed parity reports. The frozen student OOD artifact SHA-256 is
`ee2baa18a5d36cda8f9d66df642b6f3c106f92ae8ce2813acca36fa9f0e4acd2`;
the student template SHA-256 is
`f104aa18fe9f04094efdca14f9cebe65f7b18d976f454c134fa4282bd5162eaa`.
The executable config is
[`../../experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json`](../../experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json),
and the completed campaign revision is
`5a36a0dc5ccf718396eb988edb0306663c4ac237`.

No OOM occurred. The long run encountered four intermittent vLLM 0.24.0
FlashInfer GDN EngineCore spins (one during 4,008-row 9B, one during mixed 9B,
and two during mixed 4B). Each was detected by a frozen log timestamp, 99--100%
reported SM utilization, zero memory utilization, and roughly 115 W power.
Exact-PID restarts resumed from durable 128-row shards without changing scores,
backend, adapter, or prompt. Future long Qwen3.5 evaluations should add a shard
watchdog and benchmark Triton GDN prefill under a fresh parity gate.

The two mixed adapters are additionally registered in
[`tool_trajectory_ood_frontier.md`](tool_trajectory_ood_frontier.md); the six
matched-size points remain here so the general frontier is not overloaded with
one experiment's internal scaling curve.
