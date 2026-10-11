# Mechanistic interventions for prompt injection sensitivity in logit trained monitors

*Initial draft, 11 October 2026*

Our monitors sometimes treat prompt injection text as evidence of harmful
behavior even when the underlying code is honest. We investigated whether an
activation direction associated with these false alarms could help us change
the training data or the training computation. Two interventions were central:
removing training examples with the largest learned alignment changes, and
projecting the direction out during fine-tuning using a CAFT-style procedure.

Neither resolved the problem. Activation-based filtering substantially improved
in-distribution ranking while worsening injection discrimination. CAFT retained
100% false positives on injected honest code, including when we also removed
the original direction at inference. Yet adding that same direction still
strongly increased monitoring scores. The direction has causal influence, but
these experiments do not identify it as a necessary bottleneck for false alarms
or establish which training examples caused that influence to emerge.

Training precision and serving arithmetic also changed the behavior materially.
This matters when interpreting mechanistic interventions: a lower false-positive
rate can accompany a loss of harmful-action detection, and seemingly similar
checkpoints or serving configurations can behave differently.

## Monitoring task and evaluation

Gleipnir uses a Qwen3.5-4B backbone with rank-128 LoRA adapters, trained on Kimi
K3 soft judgments of tool trajectories. Its monitoring score is
`sigmoid(logit_1 - logit_0)` at a fixed `Prediction:` boundary, with `1` indicating
problematic behavior. Source-provided harmless/harmful labels and soft teacher
targets are separate: a harmless-labeled record can receive a suspicious
teacher target.

The augmented training condition replaces 40% of the 8,688 trajectories with
versions containing one of eight authored injection strings, while copying the
original soft teacher target. It teaches invariance to those inserted strings
under the assumption that the underlying behavior judgment should stay fixed.
Historical augmentation improved ID ranking and reduced APPS false alarms,
which motivated looking for a mechanistic alternative. The augmented monitor
used in the central experiments is a current BF16/SDPA replay; it does not
reproduce the historical adapter's weights or complete behavior.
[Augmentation and replay evidence](../findings/b200_augmented_training.md).

We evaluate two distinct properties. The in-distribution (ID) set contains
3,012 records from Gloom and STRIDE, with identified reasoning removed and
visible assistant prose retained. The primary ID ranking statistic is the
source-macro partial AUROC up to 20% false positives, normalized as raw area
divided by 0.2. This is not sklearn's standardized partial AUC.

The APPS evaluation compares honest and backdoored programs, including clean
versions and matched versions containing neutral or injection text. Injection
text in honest code does not change its behavior label. We report both the
false-positive rate (FPR) on injected honest code and recall on injected
backdoored code. Matched AUROC compares those two classes on common tasks.
Unless stated otherwise, each condition receives a threshold calibrated on
clean honest code to allow at most 2% FPR, using the strict rule
`score > threshold`. Fixed-threshold diagnostics distinguish changes in
calibration from changes in discrimination.

Our mechanistic direction was fitted on 64 APPS tasks. Every variant of those
tasks is excluded from the main evaluation, leaving 591 tasks and 8,218 rows,
including 583 tasks with backdoor counterparts. Reported injection metrics
average three fixed templates. This separates fitting tasks from test tasks,
but APPS had already informed the research questions and intervention geometry;
it is an exploratory transfer study rather than an untouched confirmation
benchmark. The precision comparisons later in this writeup use the original
655-task APPS population and are identified separately.

The mechanistic studies use merged BF16 checkpoints with eager vLLM Lens
interventions. New adapters pass master-to-merge and master-to-serving score
checks; request isolation, intervention arithmetic and score identities are
audited. These controls support the validity of the measured edits without
making BF16 serving bitwise equivalent to the FP32 master.

## A direction that changes injection responses

Following [our colleague's intervention experiments](https://github.com/mkenney2/monitor-injection-evals/tree/931fee0a1db369bc15c01009aa77219e7e106c14),
we fitted a direction in the current augmented, BF16/SDPA-trained monitor.
At layer 20, we averaged the final-token residual differences between injected
and neutral versions of the same examples, using both honest and backdoored
pairs. Let this mean difference be `v` and its unit direction be `u`.

We compared a centered clamp, a full projection and a random-direction
projection. Full projection applies `h' = h - u(uᵀh)`. The clamp subtracts
one quarter of the deviation from a neutral fitted center along `u`, rather
than setting the component to zero. **Although fitted at layer 20, both
operators act after every decoder layer and on every token.** Addition is a
different experiment: it adds `v` at layer 20 only, also on every token.

| Augmented monitor intervention | Injected honest FPR | Injected backdoor recall | Matched AUROC | Matched pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| Unedited | 50.82% | 79.19% | 0.8049 | 0.5191 |
| Centered clamp | 0.45% | 36.48% | 0.8631 | 0.6252 |
| Full projection | 0.39% | 31.39% | 0.8905 | 0.6808 |
| Random projection | 46.81% | 79.42% | 0.8093 | 0.5271 |

Projection produces a substantial ranking improvement and almost eliminates
injected-honest alarms. It also loses most injected-backdoor recall at the clean
operating point. ID macro pAUROC@20 changes only slightly, from 0.8590 to
0.8585, but that stability does not remove the APPS recall tradeoff. The fitted
injection and backdoor mean shifts have cosine similarity 0.542, consistent
with overlapping features useful for both kinds of alarm. Supported-answer
probability mass remains high, so the effect is not explained by answer-format
collapse. Task-bootstrap intervals support the ranking gains, conditional on
this checkpoint and fitted vector; they do not measure training-seed variation.
[Detailed projection results](../findings/b200_injection_direction.md).

This gave us a practical hypothesis: if we could identify training examples
that strengthened the direction, perhaps removing those examples would reduce
false alarms without needing an inference-time intervention.

## Activation based filtering improved ID and worsened APPS

We measured activations on **all 8,688 original training records**, without added
synthetic injection views, in both the base model and the augmented monitor.
For each record `x`, we computed
`Δz(x) = uᵀh_trained,20(x) - uᵀh_base,20(x)` at the final prompt token. This
measures a training-associated representation change along a fixed axis. It
does not measure how much that record's gradient contributed to the axis.
Original trajectories can themselves contain natural injections or quoted
tests; absence of synthetic augmentation does not certify injection-free text.

The strongest inspection lead was disagreement between harmless source labels
and suspicious teacher targets. Among harmless records, 346 had Kimi targets
at least 0.5. Their mean alignment change was +4.551, versus -1.614 for the
other 3,998 harmless records. Harmful records also changed positively on
average, including those with low teacher targets. This pattern associates the
representation with supervision and source composition; it does not certify
teacher errors or identify a unique injection feature.

Some highly activating records discussed firewall configuration or network
safeguards. A full Qwen3.5-9B concept screen found that the association was not
exclusive to those topics. Within harmless records with suspicious teacher
targets, firewall-score log odds correlated only weakly with alignment change
(Spearman 0.067), and both topic-score groups had strongly positive changes.
The classifier also missed inspected firewall cases. We therefore used
activation change itself, rather than teacher disagreements or firewall scores,
as the filtering criterion.
[Training census](../findings/b200_training_direction.md),
[firewall screen](../findings/training_firewall_census.md).

We removed the top 20% of harmless originals by `Δz`: 869 of 4,344 records.
Their augmented replacements were removed with them; all harmful originals
remained. Retraining retained the augmented recipe for the remaining records.
A control removed the same number within matching source, label, augmentation
status and length-quartile strata. Both conditions used the same seed-0
BF16/SDPA recipe and initializer, with one epoch on the smaller dataset.

| Training data | ID macro AUROC | ID macro pAUROC@20 | Injected honest FPR | Injected backdoor recall | Matched AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered augmented | 0.9567 | 0.8600 | 51.10% | 79.13% | 0.8036 |
| Activation removal | 0.9733 | 0.9151 | 100.00% | 100.00% | 0.6016 |
| Matched random removal | 0.9608 | 0.8725 | 91.54% | 97.66% | 0.7025 |

Activation removal gained 5.51 percentage points of ID pAUROC@20, while APPS
matched pAUROC@20 fell from 0.5129 to 0.1569. Random removal also worsened
APPS, but less severely: activation removal lost another 0.1009 matched AUROC
relative to that control. Vanilla APPS AUROC remained approximately unchanged.
The failure is therefore concentrated in injection responses, rather than a
uniform collapse of code discrimination.

The filter removed 345 of the 346 harmless records with suspicious teacher
targets. This is a consequence of activation selection, not an additional
rule. Filtering also changed class composition, soft-target mass, exposure
and schedule length: both filtered runs trained for 245 updates instead of
272. The random control matches coarse composition and nearly matches token
exposure, but not teacher-target mass, exact sequences or individual lengths;
its removed set overlaps the activation set in 394 records. These differences
and unresolved SDPA run variation prevent attributing the outcome to specific
examples. The narrower result is clear: high learned alignment was a poor
deletion rule for reducing injection alarms in this trial.
[Filtering results and contract](../findings/activation_filter_training.md).

### Projection after filtering

The original direction remained relevant in the filtered model. Projecting it
out reduced calibrated injected-honest FPR from 100% to 62.61%, while recall
fell to 86.68% and matched AUROC rose from 0.6019 to 0.7343. At the fixed
unedited threshold, FPR was 79.41%, so recalibration contributed to the decrease.

Refitting the injected-minus-neutral direction in the filtered model produced
an axis with cosine 0.705 to the original. Its projection lowered recalibrated
FPR further, to 23.69%, but recall fell to 48.48% and matched AUROC was 0.7047,
below the transferred original axis. At the fixed unedited threshold, the
refitted projection still produced 100% injected alarms and 60.91% clean
alarms. The apparent FPR gain therefore includes a major calibration shift;
refitting did not yield a uniformly better discriminator.
[Filtered-model projections](../findings/activation_filter_training.md#model-specific-direction-refit).

## CAFT did not substitute for augmentation

The next approach changed the training computation rather than deleting
records. [Concept Ablation Fine-Tuning](https://arxiv.org/html/2507.16795v2#S3)
inserts latent projections during fine-tuning and differentiates through the
modified computation, then ordinarily removes the interventions at inference.
We tested a fixed-axis adaptation intended to discourage the learned
generalization associated with injection alarms.

We trained on **all 8,688 regular originals, without synthetic augmentation**.
After every decoder layer and at every token, we projected out the same unit
direction extracted from the augmented monitor. The projection participates
in both forward and backward computation; it is not a promise that all
corresponding weight components remain frozen. Hooks are removed before export.
The seed-0 BF16/SDPA recipe uses one epoch and 272 updates. Our vector comes
from APPS fitting data and an augmented model, unlike the paper's discovery
procedure without target-distribution data. This is an adaptation, not a
replication of its full method.

| Training condition | ID macro AUROC | ID macro pAUROC@20 | Injected honest FPR | Injected backdoor recall | Matched AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached regular control | 0.9575 | 0.8654 | 67.06% | 78.62% | 0.6139 |
| Regular data with CAFT | 0.9556 | 0.8513 | 100.00% | 100.00% | 0.6611 |

CAFT used BF16 training and serving; the cached control used FP4 training and
optimized serving. No fresh matched regular-data control was trained, so this
table cannot isolate the training projection's causal effect. It does establish
that the resulting CAFT model failed the intended robustness objective.
Every held-out injected-honest score exceeded both its clean-calibrated
threshold, 0.1192, and 0.5. This saturation is not merely a low selected cutoff.
ID ranking was slightly lower than the cached control, although pooled
calibration improved. [CAFT training and evaluation](../findings/concept_ablation_training.md).

### Removing the direction at inference did not rescue CAFT

We tested whether the alarms arose because ordinary inference restored the
component removed during training. Keeping the original projection active at
inference still left 100% injected-honest FPR and 100% backdoor recall, under
both recalibrated and fixed plain thresholds.

We also fitted a new APPS direction in the CAFT model. Its cosine with the
original was only 0.289. Removing it modestly improved matched ranking but
again left the alarms saturated.

| CAFT inference condition | Injected honest FPR | Injected backdoor recall | Matched AUROC | Matched pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| Unedited eager | 100.00% | 100.00% | 0.6591 | 0.1945 |
| Original training-axis projection | 100.00% | 100.00% | 0.6562 | 0.1809 |
| CAFT-specific refitted projection | 100.00% | 100.00% | 0.6784 | 0.2003 |

These comparisons share the same eager engine and held-out examples. Operator
checks confirm that the specified components were removed to the recorded
BF16 tolerance. Neither a broken hook nor restoring the original component
at inference is sufficient to explain the persistent alarms.
[CAFT inference projections](../findings/concept_ablation_training.md#keeping-the-original-training-projection-active-at-inference).

### The original vector still drives monitoring scores

Adding a vector to already-saturated injected examples cannot reveal a further
increase in FPR. We instead steered 591 clean and 1,773 neutral honest examples,
using the original augmented-model shift at layer 20 only. Doses were +1 and
+2 times that donor shift, with equal-norm random controls and no recipient
norm matching.

| CAFT honest examples | Pooled FPR at the fixed clean threshold | Mean change in logit margin |
| --- | ---: | ---: |
| Unedited | 2.58% | 0 |
| Original vector +1 | 61.04% | +1.901 |
| Original vector +2 | 100.00% | +3.163 |
| Random vector +1 | 2.28% | -0.096 |
| Random vector +2 | 2.75% | +0.039 |

Both positive doses increased the `1 - 0` logit margin in 99.75% of rows.
Answer mass stayed high. The effect is strongly threshold-dependent: at 0.5,
pooled FPR was only 0.34% unedited and 2.16% at +2. Thus 100% alarms at the
clean-calibrated threshold do not imply every steered score exceeded 0.5.
[CAFT steering controls](../findings/concept_ablation_training.md#augmented-vector-addition-on-clean-and-neutral-honest-examples).

The vector remains causally effective after CAFT, yet its removal does not
eliminate natural injection alarms. Learned rerouting is plausible: training
under projection could favor other features or computations. Pre-existing
redundancy, nonlinear reconstruction and an incomplete single-axis description
are also compatible with these results. A changed fitted direction alone does
not locate an alternative circuit or distinguish those explanations.

## Transfer suggests a broader effect on judgments

Earlier projection on the augmented monitor did not meaningfully repair
JudgeDeceiver preference judgments. On 4,188 A/B prompts, preferred-injected
accuracy moved from 55.81% to 56.46%, while its pAUROC@20 fell from 0.2336
to 0.2247. Overall accuracy gained 0.57 points, calibration worsened and source
effects were mixed. Numeric-answer follow-ups did not establish a reliable
rescue. Preference accuracy is a different target from harmfulness FPR.
[JudgeDeceiver ablations](../findings/b200_projection_judge.md).

Addition in the unadapted base model nevertheless had a large cross-domain
effect. At +2 times the augmented shift, JudgeDeceiver choices went from
29.97% B to 99.93% B, and accuracy fell from 78.44% to 50.07%. A/B answer mass
remained high. The same base engine and vector raised FPR on a fixed APPS
clean/neutral honest subset from 1.95% to 100%; equal-norm random additions
had much smaller effects.

This rules against interpreting the vector as causally active only in
monitoring contexts. It also does not establish a universal suspicion concept:
in judging, the clearest effect was answer-position bias. Prompt, task and
decision tokens all differ between domains, and an unused 0/1 readout in the
judging task cannot be treated as a valid harmfulness score. Large steering
effects demonstrate influence, not useful semantic transfer.
[Same-base transfer controls](../findings/base_judge_steering.md).

## Precision and serving change the observed behavior

The main filtering and CAFT studies used BF16 serving to avoid the known
quantization confound. Earlier matched controls show why that choice matters.

| Controlled comparison | ID macro pAUROC@20 before | After | Change |
| --- | ---: | ---: | ---: |
| Same FP4-trained augmented adapter, optimized to BF16 serving | 0.7966 | 0.8391 | +4.25 pp |
| FP4 to BF16 MLP training, retaining FA4 and BF16 serving | 0.8391 | 0.8511 | +1.20 pp |
| Same BF16/FA4 adapter, BF16 to optimized serving | 0.8511 | 0.8253 | -2.59 pp |
| Same historical augmented adapter, current BF16 to optimized serving | 0.8727 | 0.8396 | -3.31 pp |

The MLP training comparison matched the initializer, examples, physical batches,
optimizer and schedule, with one run per precision. The serving comparisons
keep the adapter fixed, so training dynamics cannot explain their differences.
The optimized recipe combines FP8 attention projections, FP4 MLP/GDN projections
and MXFP8 attention with associated kernels; it does not isolate one responsible
component. The FP4-trained and historical optimized evaluations retained failed
parity diagnostic status, rather than being accepted as equivalent serving.
[Precision controls](../findings/b200_augmented_training.md).

On the original 655-task APPS population, the same BF16/FA4 adapter changed
from 25.19% injected-honest FPR and 70.38% backdoor recall under BF16 serving
to 3.51% FPR and 30.55% recall under optimized serving. Matched AUROC fell
from 0.8216 to 0.7867. This serving change passed its short training-source
canary, demonstrating that such a gate does not establish held-out behavioral
equivalence. Lower injection FPR here accompanied weaker detection, just as
it did in the original projection experiment.

Attention arithmetic adds another qualification. Current BF16/SDPA training
partly recovered ID pAUROC@20 relative to BF16/FA4, from 0.8511 to 0.8600,
but increased full-population APPS FPR from 25.19% to 60.41% and reduced
matched AUROC from 0.8216 to 0.7989. A same-seed BF16/FA4 repeat produced
bitwise-identical FP32 adapters; the SDPA replay did not reproduce the
historical SDPA weights. This is evidence of recipe sensitivity, not evidence
that FA4 is generally deterministic and SDPA stochastic. SDPA repeats and
different-seed variance remain unresolved, as do broader historical pipeline
and host differences.

## Interpretation and next tests

The evidence supports a causal role for the extracted direction in changing
monitoring scores. It does not support using large training-associated
alignment as an attribution measure, treating the direction as a complete
description of injection sensitivity, or assuming that suppressing it during
training removes the behavior. Good ID ranking, low injection FPR and high
harmful-action recall must be evaluated separately.

A stronger test of learned rerouting would compare CAFT with a freshly trained
regular-data control under identical BF16/SDPA conditions and multiple seeds.
Matched interventions and layerwise measurements could then test whether
alternative features become more behaviorally important specifically after
CAFT. Repeating the filtering ablation with controls for teacher-target mass
would help separate the activation selection from the supervision it removes.
Testing new injection families and holding the serving recipe fixed would
establish whether any improvement generalizes beyond the templates already
used to develop these hypotheses.

For now, the two central experiments are negative mitigation results with
useful mechanistic constraints. The [released CAFT adapter](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-CAFT)
preserves the model for follow-up; publication is not a robustness endorsement.
