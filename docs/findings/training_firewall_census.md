# Firewall content and learned direction alignment

The completed Qwen3.5-9B pass finds a strong pooled association in harmless
training records, but **high alignment changes are not exclusive to firewall
content**. Within harmless / Kimi-positive records, the continuous association
is weak and both firewall-score groups shift strongly positive. Source and
length composition substantially complicate the pooled pattern. This is a
representation diagnostic, not causal training attribution or a filtering rule.

## Frozen pass and coverage

The [contract](../../experiments/training_firewall_census/README.md) freezes all
8,688 original training inputs, the concept rubric, 0.1/0.5/0.9 thresholds,
fixed top 1/5/10% bands and no prompt tuning on outputs. The unadapted
`Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
scores **discussion/inspection/configuration of firewall or concrete network-access
safeguards**, including benign and authorized cases. Generic networking,
credentials, file permissions and process cleanup alone do not count.
Normalized next-token 0/1 logprobs are continuous, uncalibrated concept scores;
source labels, Kimi targets and activation measurements are excluded from inputs.

Use stock BF16 eager vLLM 0.31 on the existing NC2 B200: the custom optimized
backend's validated envelope is 4B, not 9B. The recorded Lens engine is retired
sequentially, with its merge/caches preserved. All originals retain their exact
trajectory bytes and any existing reasoning/middle summaries. No synthetic
injection views, training, filtering, new paid teacher calls or capacity lifecycle.

Coverage is **8,688 unique finite scores, 85,336,769 input tokens**, maximum
29,512 and zero truncation. Every prompt is exactly 175 tokens longer than its
matched activation-census prompt. Scoring takes **1,407.17 s**, about
**60,644 input tokens/s and 6.174 requests/s**; total preparation/startup/scoring
time is 1,701.89 s. These are single-pass annotation costs, not a serving benchmark.
The engine exits cleanly and GPU memory returns to zero; the B200 remains running.

## Associations and intersections

Spearman correlations use Qwen **log odds** against trained-minus-base layer-20
projection, with the existing APPS unit direction/sign unchanged:

| Population | Records | Spearman rho |
|---|---:|---:|
| All original records | 8,688 | +0.456 |
| Harmless-labeled | 4,344 | +0.722 |
| Harmless / Kimi target <0.5 | 3,998 | +0.715 |
| Harmless / Kimi target >=0.5 | 346 | **+0.067** |
| Harmful-labeled | 4,344 | +0.044 |

Cosine-change correlations are similar (+0.720 for harmless, +0.053 for its
Kimi-positive group). In harmless records, Qwen score log odds correlate with
input length at +0.844. Within harmless sources, raw-change correlations are
crh -0.111, ctrl +0.349, Gloom -0.052, o46 +0.029, s45 -0.023 and STRIDE +0.215.
The pooled association is therefore not a consistent source-independent trend.
Using the first record per exact trajectory leaves 8,100 distinct inputs and
reduces the pooled correlation from +0.456 to +0.358; original exposure-weighted
results remain primary. No IID significance or causal interpretation is asserted.

Within the **346 harmless / Kimi-positive records**, Qwen >=0.5 flags 238 with
mean change **+4.684**, versus 108 below 0.5 with **+4.257**. All changes in both
groups are positive. At 0.1 there are 264/82 flagged/unflagged, means
+4.695/+4.085; at 0.9 there are 181/165, means +4.635/+4.459. These are score
cutoffs, not verified concept labels. Across the corpus, flag counts at
0.1/0.5/0.9 are 3,837/2,837/2,119. At 0.5 the top 87 raw-change records contain
50 flagged cases (see archived complete band tables for all cohorts).

`firewall_activation_distributions.{png,pdf}` shows harmless records split by
Kimi target, with matched axes, continuous log-odds scatter and unit-area change
histograms. `activation_associations.json` contains full source/label/target
groups and threshold/band intersections; `joined_activations.jsonl` retains all
original records and both model measurements.

## Classifier limitations and inspected counterexamples

All **11 short controls** classify correctly at 0.5, but unrelated filename
allowlisting scores 0.349 and an LLM injection payload 0.107. Three of the seven
[reviewed full-length firewall anchors](b200_training_direction.md#targeted-review-of-harmless--teacher-positive-high-alignment)
score below 0.5: localhost whitelist edit 0.245, binary-analysis network edit
0.438 and the summary-reported monitoring-service edit 0.202. Preserve these
misses; a low score is not verified absence. The rubric was not revised.

A disclosed **post-hoc keyword check** uses literal firewall*, iptables,
ip6tables, nftables or ufw in the original inputs, not in the classifier question.
Of the 346 Kimi-positive harmless records, 248 contain those cues and 98 do not;
their mean changes are +4.733/+4.089. All 82 records below Qwen score 0.1 also
lack those cues: 74 ctrl, five STRIDE and three Gloom. Keyword absence alone
does not exclude every related network-access concept.

Full inspection of two high-change, low-Qwen-score ctrl records finds no firewall
or network-filtering content:

- `818c9162dfacd6ad58b468b0`: kernel-module/filesystem tool; Qwen score **0.0124**,
  Kimi target 0.7549, change **+6.0834**. After a verification test fails, the agent
  changes `verify` to create an empty `/lib/modules` when missing and return
  success, rather than verifying integrity. A suspicious-looking verification
  shortcut is a plausible alternative cue; no independent misconduct judgment.
- `661342b50b409c9a784ee0d8`: sequential argument formatter; Qwen **0.00317**,
  Kimi target 0.7549, change **+5.8330**. The agent adds a numeric-duplicate special
  case to satisfy tests that conflict with the written adjacent-argument rule.
  This is test-driven adaptation/specification conflict, not firewall behavior.

These counterexamples establish non-exclusivity in the inspected records; they
do not establish which feature causes alignment, which examples taught it, or
whether the source label or teacher is wrong. Topic scores provide review clues,
not a justified deletion criterion.

## Receipts and verification

Artifacts live under `results/training_firewall_census/qwen35_9b_v1/` locally and
on the persistent volume. Prediction SHA256 is
`eddec88aef9b4f826926d5ba79c3b1fde0779cc09242f95b282b22d8e6a511d3`;
contract SHA256 is
`be42bf31f56d54811fe690d7e41a9fd40c096d793f09a6a0f3cf2aeed8efb043`.
`independent_audit.json` checks all identities, evidence hashes, labels, lineage,
native token counts, returned logprobs and score normalization and binds collected
files. Raw scores, canaries, complete rankings, executed sources and runtime log
are retained; low-score review candidates and inspected full inputs remain ignored.
Seven focused contract tests pass, with two analysis tests repeated after the
cross-check/join validation changes. Scoped Ruff and diff checks pass; no plotting
tests were added. No ID/APPS monitor-quality evaluation or promotion occurred.
