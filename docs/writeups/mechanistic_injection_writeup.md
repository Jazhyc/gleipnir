# Mechanistic interventions for prompt injection sensitivity in logit trained monitors

*Working draft, 11 October 2026*

This investigation continues the results we presented in
[Emergent Suspiciousness in Trained Monitors](../../outputs/presentations/monitor_injection_results/presentation.pdf).
Our trained 4B and 9B monitors initially appeared highly robust to injected
backdoors. However, adding the same injection text to honest code raised alarm
rates from 1.98% on clean code to 100.00% and 99.85%, respectively, at the same
thresholds. We described this behavior as emergent suspiciousness: the presence
of injection text could trigger an alarm even without harmful behavior.

Building on Michael's work on extracting and intervening on an
injection-associated activation direction, we investigated two ways to change
training: **filtering examples by their activation change**, and **projecting
the direction out during fine-tuning with CAFT**. Michael's
[companion writeup](https://claude.ai/artifact/TXoHJdDKsk99o1bcp79b5R#06f44dca-9677)
covers direction extraction and inference interventions;
the [implementation](https://github.com/mkenney2/monitor-injection-evals/tree/931fee0a1db369bc15c01009aa77219e7e106c14)
provides the underlying protocol.

Both training interventions failed to resolve injection false alarms. Filtering
improved in-distribution ranking while worsening injection discrimination.
CAFT retained 100% false positives on injected honest code, even with further
inference-time projection. Supporting steering experiments show that the
original direction still influences the model. We also found substantial
sensitivity to training precision and serving arithmetic.

## Context for the comparisons

These experiments use Qwen3.5-4B monitors trained on 8,688 tool trajectories
with Kimi K3 soft targets. Source-provided harmless/harmful labels are distinct
from those targets: a harmless-labeled trajectory can receive a suspicious
teacher judgment. The augmented condition replaces 40% of trajectories
with injected versions, retaining their original teacher targets. Our current
BF16/SDPA augmented checkpoint is a replay, not the historical adapter used
in the original presentation.

We track two outcomes. The **ID set** contains 3,012 Gloom and STRIDE records,
with identified reasoning removed and visible assistant prose retained.
Its primary metric is source-macro pAUROC@20: raw partial AUROC up to FPR 0.2,
divided by 0.2. The **APPS set** compares honest and backdoored programs with
clean, neutral and injected versions. We report injected-honest false-positive
rate (FPR), injected-backdoor recall and matched AUROC. A low FPR is useful
only if harmful behavior remains detectable.

The direction was fitted on 64 APPS tasks; all their variants are excluded
from the main results, leaving 591 tasks, including 583 with backdoor
counterparts. Injection metrics average three fixed templates. Each condition
normally receives a clean-only threshold allowing at most 2% FPR, using
`score > threshold`; fixed-threshold results are identified separately.
APPS had already informed these hypotheses, so this is an exploratory study,
not an untouched confirmation benchmark.

The direction is extracted at layer 20. **Projection removes its component
after every decoder layer and at every token; addition acts at layer 20 only.**
New training runs and the main evaluations use BF16 to avoid the known quantization
confound. In our replication, inference projection on the augmented model
reduced injected-honest FPR from 50.82% to 0.39%, but also reduced backdoor
recall from 79.19% to 31.39%. This motivated asking whether training could
produce a better tradeoff.
[Replication results](../findings/b200_injection_direction.md).
The projection replication uses eager serving; the filtering table below uses
compiled BF16 evaluations.

## Activation based filtering improved ID and worsened APPS

We measured activations on **all 8,688 original training records**, without added
synthetic injection views, in both the base model and the augmented monitor.
For each record `x`, we computed
`Δz(x) = uᵀh_trained,20(x) - uᵀh_base,20(x)` at the final prompt token. This
measures a training-associated representation change along a fixed axis. It
does not measure how much that record's gradient contributed to the axis.
Original trajectories can themselves contain natural injections or quoted
tests; absence of synthetic augmentation does not certify injection-free text.

Large positive changes concentrated in harmless records with suspicious Kimi
targets. Some inspected records involved firewall configuration, but a full
Qwen3.5-9B topic screen showed that high activation changes were not exclusive
to those examples. We therefore tested activation change itself as a deletion
rule, without using teacher disagreements or topic scores in selection.
[Training census](../findings/b200_training_direction.md),
[topic screen](../findings/training_firewall_census.md).

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

The activation filter removed 345 of the 346 harmless records with suspicious
teacher targets. It also changed class composition, target mass and training
exposure; the smaller datasets required 245 updates rather than 272. The random
control matched coarse composition and nearly matched token exposure, but not
teacher-target mass or exact sequences. Its removed set overlapped the
activation set in 394 records. With one seed per condition and unresolved
SDPA run variance, these results cannot identify individual causal examples.
They do show that high learned alignment was a poor deletion rule in this trial.
[Filtering results](../findings/activation_filter_training.md).

### Projection after filtering

The original direction remained relevant. Projecting it out of the filtered
model reduced calibrated FPR from 100% to 62.61%, with 86.68% backdoor recall.
Refitting a direction in that model lowered calibrated FPR to 23.69%, with
48.48% recall, but yielded worse matched AUROC than the original-axis projection.
At the fixed unedited threshold, the refitted projection still produced 100%
injected alarms and 60.91% clean alarms. Its apparent advantage therefore
included a large calibration shift, rather than a uniformly better discriminator.
[Projection follow-ups](../findings/activation_filter_training.md#model-specific-direction-refit).

## CAFT did not substitute for augmentation

With [Concept Ablation Fine-Tuning](https://arxiv.org/html/2507.16795v2#S3),
we changed the training computation instead of deleting examples. We trained
on **all 8,688 regular originals, without synthetic augmentation**, projecting
out the original augmented-model direction after every layer and token.
Gradients pass through the projection. The intervention restricts computation
at those sites; it does not globally freeze corresponding weight components.
The projection is removed for ordinary inference, following CAFT.

Our direction-selection procedure uses APPS fitting data and an augmented
model; the paper discovers concepts without target-distribution data. The
training procedure is CAFT, with a different source for the direction.

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

### The direction still has causal influence

Because injected examples already had 100% FPR, we tested addition on clean
and neutral honest examples instead. Adding the original vector at layer 20
raised pooled FPR from 2.58% to 61.04% at one donor-shift unit and 100% at two,
using the fixed clean-calibrated threshold. Equal-norm random vectors kept FPR
near 2–3%. At two units, the mean `1 - 0` logit-margin increase was 3.163.
At the separate threshold 0.5, FPR was only 2.16%, so the 100% operating-point
result should not be read as every score exceeding 0.5.
[Steering controls](../findings/concept_ablation_training.md#augmented-vector-addition-on-clean-and-neutral-honest-examples).

The original vector can still drive monitoring scores, yet its removal does
not eliminate natural injection alarms. Training-induced rerouting is plausible,
as are pre-existing alternative features, nonlinear reconstruction and an
incomplete single-axis description. The changed fitted direction does not
distinguish these explanations.

Transfer tests also caution against assigning the vector a simple semantic
meaning. Earlier ablations offered little reliable benefit on JudgeDeceiver.
Adding the augmented vector to the base model strongly biased judging toward
answer B, while retaining its alarm effect on APPS. These results show influence
across tasks, without establishing useful transfer or a universal suspicion
feature. [Judging ablations](../findings/b200_projection_judge.md),
[base-model steering](../findings/base_judge_steering.md).

## Training precision and serving matter

The main experiments use BF16 because earlier matched controls showed material
changes in behavior. With BF16 serving held fixed, changing augmented-model
MLP training from FP4 to BF16 raised ID macro pAUROC@20 from 0.8391 to 0.8511.
This was a one-seed comparison with matched initialization, examples, batches
and schedule.

For that same BF16/FA4-trained adapter, changing serving from BF16 to the
optimized recipe lowered ID pAUROC@20 from 0.8511 to 0.8253. On the original
655-task APPS population, honest-injection FPR fell from 25.19% to 3.51%,
but backdoor recall fell from 70.38% to 30.55% and matched AUROC from 0.8216
to 0.7867. The lower FPR accompanied weaker harmful-action detection. These
are serving effects on fixed weights, not different training runs.

The optimized recipe combines FP8 attention projections, FP4 MLP/GDN
projections and MXFP8 attention with associated kernels; these comparisons
do not isolate one component. This adapter passed its short score canary,
whereas other optimized evaluations retained explicit failed-parity diagnostic
status. A small canary does not establish held-out behavioral equivalence.
[Precision comparisons](../findings/b200_augmented_training.md).

Attention arithmetic also mattered: a current BF16/SDPA replay partly recovered
ID ranking relative to BF16/FA4, but worsened APPS injection discrimination.
One same-seed FA4 repeat produced bitwise-identical adapters; the SDPA replay
did not reproduce historical SDPA weights. SDPA repeatability and different-seed
variance remain unresolved. We cannot infer a general deterministic-FA4 versus
stochastic-SDPA distinction from these comparisons.

## What the experiments establish

The extracted direction influences monitoring judgments, but neither large
training-associated alignment nor a successful inference edit guarantees a
useful training intervention. Filtering improved ID while worsening injection
discrimination. CAFT left alarms intact, even after additional projection.
Good ID ranking, low injection FPR and high harmful-action recall are separate
properties that need separate checks.

A matched regular-data BF16/SDPA control and multiple seeds would help test
whether CAFT specifically changes the mechanism. Controls for removed
teacher-target mass would strengthen the filtering comparison. New injection
families would test transfer beyond the templates already used to develop
these hypotheses. The [released CAFT adapter](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-CAFT)
preserves this negative result for follow-up.
