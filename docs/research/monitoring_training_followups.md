# Possible monitoring training follow-ups

Date: 2026-09-08. Status: research backlog documented at the user's request.
These are proposed experiments, not a frozen campaign, selected recipe, or
authorization to launch training or obtain additional supervision.

## Scope and working recommendation

Keep the backbone fixed and reuse the existing teacher data. Qwen3.5-4B is the
immediate target; earlier 9B deception results supply supporting evidence, not
matched estimates of effects on 4B monitoring. Retain Kimi soft-label binary
cross-entropy (BCE) as the core objective and the direct binary-logit deployment
interface. Cheap supervision is an optional extension, preferably for evidence
and authorization judgments on existing examples.

Prioritize training from the intended visible inputs, consistency under changes
that preserve evidence, and compact evidence supervision. Test checkpoint
averaging alongside these runs. Investigate exposure and data-mixture effects
with separate controlled comparisons. None is an established OOD improvement.

## Evidence motivating the backlog

Historical ID numbers below use the original 3,012-row input, which contains
identified agent reasoning. They are not baselines for the new CoT-removed
validation contract. All recent monitoring training comparisons are single-seed
development results unless otherwise stated.

| Observation | Evidence | Implication and limit |
| --- | --- | --- |
| Training duration interacts with learning rate | At `2e-5`, full-data ID pAUROC@20 rises from `0.871066` at one epoch to `0.882952` at two; the two-epoch `1e-5` run falls to `0.854856`. | Exposure matters, but longer training is not uniformly better or worse. |
| The data curve is non-monotonic | One-epoch ID pAUROC is `0.886670` at 20%, `0.833644` at 50%, and `0.871066` at 100%. | Subset content, update count, schedule and seed remain confounded; 20% is not a demonstrated optimum. |
| Stronger ID ranking has not established stronger OOD ranking | Selected 20%-data, three-epoch MIL reaches `0.895951` ID pAUROC, but `0.781299` OOD pAUROC versus released 4B's `0.782350`. | The aggregate OOD result is effectively unchanged; mixture, duration and objective differ together. |
| Adding deception can interfere under the tested recipe | Adding all 13,149 deception rows to the selected MIL recipe reduces ID pAUROC to `0.876761`. | This also changes updates, schedule context and execution; it does not isolate a harmful effect of deception content. |
| Available information differs between training and deployment | Confirmed reasoning occurs in 1,340/8,688 monitoring training examples. CoT-only removal changes released 4B ID pAUROC from `0.869194` to `0.850083`. | Train on the intended inputs. The audit is a lower bound on training reasoning exposure; the ablation is not a retraining result. |

Sources: [duration](../../experiments/monitoring_duration/README.md),
[data scaling](../../experiments/monitoring_id_scaling/README.md),
[subset duration](../../experiments/monitoring_subset_duration/README.md),
[MIL OOD protocol](../../experiments/monitoring_mil_ood/README.md),
[mixed MIL protocol](../../experiments/monitoring_mil_mixture/README.md),
[training-input audit](../findings/training_input_reasoning.md), and
[ID input ablations](../findings/id_assistant_prose_and_reasoning.md).
The completed MIL OOD aggregate is in ignored artifact
`results/monitoring_mil_ood/ood_evaluation/4b/adapters/mil-pct020-lr2em05-epochs3-seed0/result.json`;
the mixed-MIL aggregate is in `results/monitoring_mil_mixture_ddp/summary.json`.

The [strict OOD input audit](../findings/ood_input_reasoning.md) found no
confirmed private-thinking leakage. The broader assistant-prose-removal ID
ablation reaches `0.802982`, but removes useful visible prose as well as CoT;
its entire decline must not be attributed to private reasoning.

## 1. Distill privileged judgments into CoT-removed student inputs

Hypothesis: requiring the student to use observable behavior during training
improves transfer to inputs without identified private reasoning.

Remove source-typed reasoning and explicit thinking calls from student inputs,
preserving visible assistant prose, other calls and arguments, tool results,
task context and reports. Compare with a fresh historical-input control using
identical parents, targets, seed and update schedule. Keep original inputs and
teacher caches immutable; create a derived student artifact with transformation
provenance and distinct hashes.

Retain the existing Kimi probabilities as privileged targets. They were computed
from richer inputs and must not be relabeled as predictions on cleaned prompts.
This is a proposed application of generalized distillation, which permits
teacher and student information to differ. It does not require new teacher
calls. See [Lopez-Paz et al., Unifying Distillation and Privileged
Information](https://arxiv.org/abs/1511.03643).

Removed reasoning may contain the only discriminating information in some
examples. A student cannot reliably reconstruct unavailable evidence; preserve
soft uncertainty and inspect calibration. Do not impose example-by-example
confidence equality between full-input and cleaned-input student predictions.
Information removal is not automatically a label-preserving invariance.

## 2. Consistency regularization and optional counterfactual pairs

Hypothesis: discouraging sensitivity to irrelevant presentation reduces reliance
on source-specific cues without obtaining additional teacher labels.

One candidate uses two evidence-equivalent views of each parent, soft BCE on
both, and a symmetric KL penalty between their binary distributions. Average
the two supervised losses so duplicating views does not double parent weight.
Freeze the consistency coefficient, transformation distribution and schedule
before training. Report the extra forward/backward cost separately.

Possible transformations include alternate serialization and consistent renaming
of incidental identifiers. Validate preservation of identity relationships,
authorization boundaries, trust status, action effects and temporal order.
Arbitrary truncation, removal of suspicious actions, or changing a destination's
trust status cannot inherit the original target automatically.

A simpler implementation candidate is R-Drop: independent LoRA dropout masks
for two passes of the same input, soft BCE on each, and symmetric KL. Nonzero
dropout is necessary for that comparison; the existing zero-dropout recipe
would provide no intended stochastic perturbation. Compare against ordinary
dropout at the same rate to isolate the consistency penalty. The earlier
dropout-only screen does not test this objective. Supporting NLP results come
from [R-Drop](https://arxiv.org/abs/2106.14448); transfer to long-trajectory QLoRA
remains untested.

With cheap supervision, a later extension could create coherent paired examples
where a verified change in authorization or behavior changes the label while
most context remains fixed. Train on both with their separately validated
targets; a pairwise objective is optional, not implied. Keep each pair and its
parent in the same split. [Counterfactual augmentation](https://arxiv.org/abs/1909.12434)
supports this direction in NLP, not specifically in agent monitoring. This
would be new derived supervision and needs its own provenance and quality audit.

## 3. Compact, grounded auxiliary supervision

Hypothesis: supervising specific evidence and authorization judgments develops
useful monitoring representations more efficiently than imitating full analyses.

Reuse the stored Gemini analyses to derive a small set of auxiliary targets:

- visible action/output spans that support the judgment;
- whether the requested task authorizes the action;
- whether the final report agrees with the observed action or result.

Check cited evidence against the actual visible student input. Mask unsupported
or ambiguous targets rather than invent labels, and distinguish locating a
cited span from verifying that its interpretation is correct. Include grounded
benign explanations and ordinary mistakes. If extraction is unreliable, a
bounded cheap-supervision extension should validate these targets on existing
examples rather than annotate every prefix again.

The stored rationales were selected or regenerated to agree with source labels.
Keep their terminal ordinal verdict out of the evidence target and separate
from Kimi probabilities. Preserve provenance for hard labels, privileged
analyses, derived evidence and soft targets.

Test a small auxiliary weight early in training, tapering it toward zero while
retaining direct soft BCE throughout. Freeze the taper and normalize auxiliary
loss within each parent and over the declared parent population, so longer
analyses or more annotations do not silently increase an example's weight.
Compare with soft-only training on identical inputs and exposures. Keep direct
scoring at inference; generated explanations are not required by this proposal.

Ordinary rationale SFT is not an untested easy win. Completed corrected summaries
read on 2026-09-08 report ID pAUROC `0.870399` and `0.873341` at rationale
weights `0.05` and `0.20`, versus soft-only `0.871066`; neither reaches the
predeclared `+0.005` ranking gate. The corresponding artifact paths are
`results/monitoring_objective_ablation_accumulation_v2/id_evaluation/4b/adapters/soft-rationale-w005/result.json`
and `soft-rationale-w020/result.json` in that same adapters directory. Preserve
the invalid original runs separately; the
[accumulation audit](../findings/monitoring_objective_accumulation.md) explains
why their configured loss weights cannot be interpreted as intended.

[Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) motivates separating
explanation supervision from prediction. [WildGuard's multitask
ablations](https://arxiv.org/html/2406.18495v2#S4.SS3) motivate learning distinct
safety judgments jointly. These are supporting analogies, not demonstrations
that the proposed evidence targets improve Gleipnir.

## 4. Control exposure and mixture effects

Hypothesis: some subset and mixture reversals reflect optimization exposure or
interference between tasks rather than an intrinsic optimum in unique rows.

Compare repeated smaller pools with broader pools under identical optimizer
update budgets and LR schedules. Use multiple subset seeds, and report unique
parents, repetitions, per-source exposure, tokens and compute. A fixed update
budget alone does not match token work. Separate subset-draw variation from
adapter initialization/data-order variation where practical.

For deception plus monitoring, consider bounded task sampling or a broad first
phase followed by monitoring-focused training with limited deception replay.
To isolate order, give the interleaved control the same total source exposures
and global schedule as the staged arm. Treat a different replay ratio as a
separate exposure intervention. Check retention of deception performance as well
as monitoring; the broader mission is not a single-source specialist.

Start with simple exposure controls before adaptive loss-based weighting.
Large losses may represent annotation mismatch or unavailable information.
Earlier deception sampling/GroupDRO screens are insufficient to establish a
monitoring improvement. [Group-DRO research](https://arxiv.org/abs/1911.08731)
also shows that minimizing worst-group training loss requires appropriate
regularization to improve worst-group generalization.

## 5. Checkpoint averaging

Hypothesis: averaging a predeclared late-training window reduces sensitivity to
the final optimization state and improves transfer with little training overhead.

Compare a final checkpoint with one fixed-window average or EMA from the same
run. Freeze the averaging method, window or decay, and checkpoint cadence
before evaluation. Avoid choosing averaging members from ID or OOD scores.
Keep training FP32 masters and pass the usual serving parity gate.

Specify the LoRA averaging convention: averaging A and B factors separately is
not equivalent to averaging the effective updates B*A. Any effective-update
averaging that increases rank or requires compression must record that change
and validate the resulting artifact. Do not silently change inference capacity.

[Weight-Averaged Knowledge Distillation](https://arxiv.org/abs/2309.11446)
reports improvements under domain shift in vision. This is a cheap motivated
test, not established evidence for Qwen monitoring or LoRA factor averaging.

## Lower-priority directions

- Further broad optimizer, adapter-variant and generic regularization sweeps:
  earlier results provide limited support for large gains. See the
  [structural screen](../../experiments/training_procedure_screen/README.md)
  and [migrated findings](../migrations/aletheia_distillation_findings.md).
- Stronger or denser prefix supervision: sampled gains were modest and
  inconsistent; the existing prefix cache has unresolved numerical limitations
  and source-dependent teacher mismatch. All-prefix branching reached only
  `0.803823` ID pAUROC at matched allocated GPU time after visiting 336 parents.
  This rejects that implementation's compute efficiency, not a converged
  all-prefix objective. See [prefix results](../../experiments/monitoring_prefix_supervision/README.md),
  [branching results](../../experiments/monitoring_branching/README.md), and
  [teacher agreement](../findings/matched_qwen_kimi_teacher_agreement.md).
- Hard-label anchoring: replicated internal deception gains did not establish
  broad external-transfer improvement. Label semantics can differ from the
  monitoring rubric. See the [external comparison](../../experiments/aisi_lie_detection_transfer/README.md).
- GRPO or generic pairwise ranking: historical RL results and prior pairwise
  overfitting reduce their priority. In the closest
  [deliberative-monitor paper](https://arxiv.org/html/2605.29601v1#S4.SS1),
  supervised training supplies the dominant gain and RL adds smaller, uneven
  improvements. Its discussion of messy benign false positives also motivates
  maintaining honest hard negatives in proposed evidence and counterfactual work.
- Length-specific penalties: the replicated inert-padding audit did not cross
  its materiality thresholds. See the [length audit](../findings/monitoring_length_shortcut_audit.md).

## Suggested first screen and selection protocol

This is a candidate design to refine and freeze before implementation or launch.

| Arm | Student input | Training objective | Primary comparison |
| --- | --- | --- | --- |
| Control | Historical monitoring input | Kimi soft BCE | Fresh reference |
| A | CoT removed, visible prose retained | Kimi soft BCE | Input intervention versus control |
| B | Same as A | Soft BCE plus one chosen consistency method | Consistency versus A |
| C | Same as A | Soft BCE plus compact evidence auxiliary | Grounded supervision versus A |

Use the full 8,688-parent monitoring pool as a defensible initial control for
subset selection. Candidate common settings are rank-128/alpha-256 QLoRA,
AdamW `2e-5`, effective batch 32, two epochs and the established linear schedule
with 3% warmup. These are starting settings, not a newly selected optimum.
Retain matched parent exposures and LR schedules, record extra work for
auxiliary passes, and evaluate one predeclared averaged checkpoint alongside
each final checkpoint. If B uses R-Drop, add its matched dropout-only control
before attributing a gain to consistency.

Use the accepted [CoT-removed ID contract](../decisions/cot_removed_id_validation.md)
for every arm, including the historical-input training control. The released
4B reference on this surface is pAUROC `0.850083`, not its original-input
`0.869194`. Freeze current input and manifest hashes in each new config.

Select by source-macro raw normalized pAUROC@20, retaining per-source AUROC,
Brier, calibration, score ties and recall at predeclared false-positive budgets.
Report the pooled operating point too; source-specific thresholds are diagnostic
unless the deployment contract supports them. A candidate exploratory gate is
`+0.005` macro pAUROC, no source loss greater than `0.01`, and no macro-Brier
regression greater than `0.005`, each against the matched control. Freeze the
gate and treatment of final versus averaged endpoints before evaluation.

Replicate the strongest intervention and its matched control across three paired
seeds before combining methods. Add development folds that hold out entire
training sources, grouping available task/conversation/generator lineage and
all derived views. Missing lineage limits must remain explicit. CoT removal
does not make the familiar ID sources an independent confirmation set. Keep
the six-source OOD suite outside all recipe, threshold and stopping choices;
evaluate a frozen candidate there for reporting, with its prior use disclosed.

Before an expensive run, create the experiment README and config with the exact
hypothesis, intervention, baselines, held-out selection rule and stop conditions.
Fail on input/target provenance drift, invalid transformations or evidence,
incorrect parent/accumulation normalization, nonfinite loss, missing required
kernels, failed longest-context preflight, serving-parity failure or incomplete
evaluation. Preserve the existing precision, kernel and artifact contracts and
keep negative results. This backlog itself makes no changes to training inputs,
teacher targets, experiment queues or the selected recipe.
