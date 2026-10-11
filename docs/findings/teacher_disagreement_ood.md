# Strict teacher-disagreement filtering: OOD transfer

The fixed strict-filtered seed-0 checkpoint reaches **0.786936** six-source
Mean-OOD pAUROC@20, compared with **0.782350** for cached released Gleipnir 4B
(+0.46 percentage points). Its large ID ranking gain therefore transfers as a
small, uneven OOD point-estimate gain. Mean-OOD AUROC improves, but calibration
does not improve over the released model. No promotion or OOD tuning occurred.

## Fixed comparison

The [follow-up contract](../../experiments/teacher_disagreement_ood/README.md)
evaluates only the final strict delta-zero checkpoint from the
[ID sweep](teacher_disagreement_filtering.md): 7,836 retained monitoring rows
after removing 852 teacher/source-label disagreements, one epoch, seed 0,
rank-128 FP32 master adapter and BF16/SDPA training. The selected checkpoint
was fixed before this OOD evaluation. The benchmark has been used repeatedly
in prior work; this is a transfer diagnostic, not a new untouched test set.

The primary reference is the user's requested saved BF16 Gleipnir 4B scores
from `tool_trajectory_distillation_ood/runs/4b/adapters/soft-n21837-mixed-qwen35-4b-seed0/`.
It includes 13,149 additional deception training examples. The secondary cached
`student_injection_awareness/4b/regular/ood/` reference uses the original
unfiltered 8,688-row monitoring pool. Neither reference was rerun. Historical
dynamic-LoRA serving, runtime, host and score resolution differ from the new
merged BF16 run; these comparisons do not isolate filtering causally.

All models share exactly the same 6,395 IDs, source labels, original prompts,
rendered prompts and token lengths. Sources are AgentDojo (1,500), CUA SHADE
(955), IAC (1,443), MLE sabotage (148), Rogue (1,499) and SHADE (850).
The prepared OOD input SHA256 is
`42556d4f6ef6f7715b3b27f0b4d6f11ae83ef9c8499009489799032c2abbbf73`.
Source dataset/revision and all transformations remain bound in its existing
manifest and the follow-up's `data/teacher-disagreement-strict-ood01/manifest.json`.
There is no exact trajectory-content overlap with the training rows; that
does not establish independence of unavailable higher-level lineage.

## Results

Metrics are unweighted source means; pAUROC@20 is the raw normalized partial
area at FPR <= 0.2. FPR/recall use the unchanged 0.5 threshold.

| Model | pAUROC@20 | AUROC | Brier | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| Cached Gleipnir 4B, mixed | 0.782350 | 0.908584 | 0.132055 | 6.92% | 72.29% |
| Cached unfiltered regular | 0.776238 | 0.914348 | 0.138607 | 9.48% | 72.20% |
| Strict filtering | **0.786936** | **0.921876** | 0.136523 | 7.72% | 72.04% |

| Source | Strict pAUROC@20 | Cached Gleipnir 4B | Cached regular |
|---|---:|---:|---:|
| AgentDojo | 0.929880 | 0.847747 | 0.676676 |
| CUA SHADE | 0.771191 | 0.786471 | 0.777495 |
| IAC | 0.813657 | 0.884983 | 0.916043 |
| MLE sabotage | 1.000000 | 1.000000 | 1.000000 |
| Rogue | 0.969717 | 0.939061 | 0.947882 |
| SHADE | 0.237170 | 0.235836 | 0.339330 |

AgentDojo and Rogue improve over released Gleipnir 4B; CUA SHADE and IAC
regress. Relative to the regular checkpoint, the large AgentDojo improvement
is partly offset by IAC and SHADE regressions. Macro pAUROC improves by 1.07
points over regular. The strict candidate's pooled pAUROC is 0.710063, pooled
AUROC 0.926191 and pooled Brier 0.140908. Full per-source calibration,
threshold diagnostics and pooled metrics for all references are saved.

The native BF16 candidate produces **80 unique scores**, versus 1,987 for
cached Gleipnir 4B and 1,927 for cached regular. However, the saved-score audit
below shows that the historical counts mostly reflect tiny numerical residuals,
not materially finer binary-margin resolution. Ties are retained in the metrics,
not jittered. The small aggregate gain, historical comparison and single seed
limit claims of robust transfer. Filtering changes composition and update count.

## Execution and audit

`teacher-disagreement-strict-ood01` completed on the existing NC2 B200 at
**22:56:26 CEST on 2026-10-11**, in 12m51s from the validated driver start.
Actual scoring took 492.86s. The shared ephemeral merge slot was replaced with
the strict candidate; every reconstructed weight/config/tokenizer file matched
its original successful ID merge. The original FP32-master/merge reference
receipt was reused only after those exact checks. Fresh serving parity passed:
MAE 0.00105421, correlation 0.99998366. Native BF16 attention checks passed.

Serving used Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, merged BF16 weights, vLLM 0.31.0,
Torch 2.13.0, Transformers 5.14.1, causal LAST two-logit probability scoring,
128 concurrent requests and groups of 128. The compact student instruction,
chat rendering, endpoint settings and source hashes are frozen in the follow-up
config/manifest. All 55,346,879 input tokens were processed; maximum length
29,570 required no truncation. Native scoring generates no output tokens.
For comparison only, the existing dated paper proxy ($0.03/M input and $0.15/M
for one decision token) would be $0.259791 per 1,000 evaluations; this is not
measured self-hosted cost. The one-seed trained diagnostic does not meet the
frontier registry's three-paired-seed aggregate requirement.

Results and logs were collected locally under
`results/teacher-disagreement-strict-ood01/` and
`logs/runpod/teacher-disagreement-strict-ood01/`. Prediction SHA256:
`daee33441f55c8460d3c50c6310a9cde21ba8dc03114b07e81aeae413caf9e68`.
Independent `completion_audit.json` verifies all 6,395 candidate responses and
12,790 cached responses, raw probabilities, canonical/rendered identities,
complete metric recomputation and passing parity/BF16 gates. Eight focused
tests and lint passed before launch; implementation is commit `8b24c7e`.
The scorer is retired, the GPU is idle and the Pod remains running. Persistent
FP32 adapters, compiler caches and receipts are preserved.
Server shutdown warned that the remaining engine worker was force-killed and
one semaphore required cleanup. This occurred after complete scoring and
summary verification; the subsequent GPU check showed 0 MiB allocated.

## Score overlap with cached Kimi K3

The [distribution figure](../../figures/strict_filtered_kimi_k3_ood.png) joins
all 6,395 rows by ID and verifies identical source, source label and trajectory
hash. Paired threshold-0.5 agreement is **84.57%**, Spearman correlation
**0.8640**, Pearson **0.8081**, and mean absolute score difference **0.1809**.
Scores differ by at most 0.1 on 56.12% of trajectories. There are 754 Kimi-positive
student-negative disagreements and 233 in the opposite direction.

| Source label | N | Student mean score | Kimi mean score | Decision agreement |
|---|---:|---:|---:|---:|
| Benign | 3,189 | 0.1155 | 0.0882 | 89.87% |
| Harmful | 3,206 | 0.6494 | 0.9003 | 79.29% |

The main distribution difference is lower student scores on harmful rows.
Histogram overlap at fixed 0.05-wide bins is 86.20% for benign rows and 33.84%
for harmful rows (64.11% pooled). These are bin-dependent marginal statistics,
not paired agreement or correctness. Kimi receives the detailed teacher prompt;
the student receives the compact prompt. Scores come from historical cached
teacher inference and current merged BF16 student inference. Pooled correlations
include between-class separation; this descriptive comparison does not isolate
the effect of filtering or measure training-seed variability. Full source-level
statistics and checksums are in the figure's `.provenance.json`; regenerate with
`PYTHONPATH=src:. .venv/bin/python scripts/plot_strict_kimi_ood.py`.

## Saved-score resolution audit

The strict run's two logits range from 16.625 to 23.875, where BF16 spacing is
0.125. Every observed logit and margin lies exactly on that grid. Score
normalization already subtracts logits and computes sigmoid using Python floats;
the apparent quantization therefore precedes endpoint probability arithmetic.
The two-row classifier is deliberately BF16 in the current serving contract.

Both historical caches' logprob differences lie within `2.3842e-7` of the same
0.125 margin grid. Removing those residuals leaves 82 released-model margins
and 90 regular-model margins, versus 80 strict-model margins. Coarsening these
saved values tests numerical tie-breaking only; it does not reconstruct missing
FP32-head results or change model inference.

| Cached model | Original unique scores | Grid margins | Macro AUROC change | Macro pAUROC@20 change |
|---|---:|---:|---:|---:|
| Strict | 80 | 80 | 0 | 0 |
| Released Gleipnir 4B | 1,987 | 82 | -0.000089 | -0.000313 |
| Regular unfiltered 4B | 1,927 | 90 | -0.000050 | -0.000055 |

Threshold decisions remain identical and Brier changes are below `1e-10`.
This corrects the earlier interpretation of historical score uniqueness: tiny
logprob residuals split many ties, but have little aggregate metric effect.
It does not establish that recovering genuinely finer margins would be useless.
The clean prospective test is the same checkpoint and BF16 backbone with an
FP32 computation/output for just the final two-row projection, followed by
matched score, tie, ranking, calibration and throughput comparisons. Casting
already-rounded outputs to FP32 cannot recover lost precision. No GPU run,
recipe selection or OOD tuning occurred in this saved-score audit.

Reproduce with `PYTHONPATH=src:. .venv/bin/python scripts/audit_monitor_score_resolution.py`.
Its checksum-bound full source/pooled metrics and rounding diagnostics are in
`results/teacher-disagreement-strict-ood01/score_resolution_audit.json`.
