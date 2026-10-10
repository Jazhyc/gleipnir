# Concept-ablation training on regular monitoring data

This regular-data CAFT run **does not reduce APPS injection alarms**: all held-out
injected-honest examples exceed the clean-calibrated threshold. ID ranking is
slightly below the cached regular control, and APPS matched ranking improves
modestly while remaining weak. This fixed-axis, one-seed trial fails the proposed
augmentation substitute; backend differences prevent attributing its regression
solely to the training intervention.

The user requested a CAFT-style alternative to synthetic augmentation: train
on all regular monitoring originals while projecting the previously fitted
injection direction out of the residual stream. The
[frozen experiment](../../experiments/concept_ablation_training/README.md)
uses one seed and the complete final checkpoint; it makes no promotion decision.

## Intervention and comparison scope

Training uses **8,688 regular originals**, all original Kimi targets and no
synthetic injection views or activation filtering. The fixed unit layer-20 axis
comes from the unfiltered augmented SDPA model, with archive SHA256
`8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7`.
It was fitted on 64 APPS tasks. Every decoder layer and token applies
`h <- h - u(u^T h)` in FP32, casting back to BF16; autograd differentiates
through this projection. This restricts gradients through these latent sites,
without globally freezing corresponding weight components.

Hooks are removed before export, the vector adds no state-dict entries, and
evaluation uses ordinary **unablated inference**. This follows the training-only
intervention in [CAFT](https://arxiv.org/html/2507.16795v2#S3); using an
APPS-fitted direction from an augmented model is our adaptation, rather than
the paper's target-distribution-free concept discovery. Training without
augmentation does not mean the direction itself is augmentation-independent.

The preserved seed-0 rank-128, zero-B FP32 initializer, BF16 MLP/SDPA/FlashQLA
profile uses batch 32, LR 5e-5, linear schedule with 3% warmup and one epoch.
The 16,384-token packing target and 29,696-token cap retain all
**83,816,369 tokens** without truncation. There are **272 optimizer updates**
and **4,550 physical microbatch forwards**; all 32 layers record exactly 4,550
projection applications. This is an explicit exception to default FP4/FA4
training and optimized serving, using BF16 serving for the new adapter.

At the user's request, existing regular-model results replace a fresh control
training run. The frozen cached control uses FP4 training and optimized serving,
with host/runtime differences. Consequently this comparison cannot isolate
the causal effect of CAFT. The augmented SDPA/BF16 model remains context,
rather than a matched regular-data control.

ID uses all **3,012 frozen CoT-removed records**. Primary APPS excludes every
variant of the 64 direction-fitting tasks: **591 tasks / 8,218 rows**, with
583 backdoor tasks. Full 9,114-row APPS summaries are diagnostic. Thresholds
follow the unchanged clean-only at-most-2%-FPR rule; partial AUROC uses raw
partial area divided by 0.2, not standardized partial AUC.

## Results

| Condition | ID macro AUROC | ID macro pAUROC20 | APPS injected-honest FPR | Injected-backdoor recall | Matched AUROC | Matched pAUROC20 |
|---|---:|---:|---:|---:|---:|---:|
| Cached regular, FP4 training / optimized serving | 0.95748 | 0.86543 | 67.06% | 78.62% | 0.61387 | 0.18646 |
| Regular CAFT, BF16/SDPA training / BF16 serving | 0.95559 | 0.85131 | 100.00% | 100.00% | 0.66112 | 0.20040 |

APPS columns average the three frozen injection templates on the primary
591-task holdout. The CAFT clean-only threshold is **0.11920**, versus
0.16452 for the control; clean FPR is 1.86% versus 1.69%. All three templates
reach 100% injected-honest and injected-backdoor alarms. Mean injected-honest
scores are **0.7893 / 0.7958 / 0.7797**, versus 0.2154 / 0.6178 / 0.1729
for the control. Every held-out injected-honest score also exceeds **0.5**
and the cached control's threshold; minima by template are 0.65135 / 0.59267 /
0.62246. Thus the 100% FPR is not just a lower clean-calibrated cutoff.
Matched template AUROC is 0.6530 / 0.6751 / 0.6553 and raw
pAUROC20 is 0.2135 / 0.1878 / 0.2000. Injection-related scores remain elevated
under ordinary inference after the training hooks are removed.

Vanilla APPS AUROC/pAUROC20 are **0.92821 / 0.74676**, versus
0.91787 / 0.74212; vanilla backdoor recall is 52.32% versus 49.23%.
Neutral-template honest FPR is 5.08% / 1.69% / 1.69%, with backdoor recall
47.86% / 38.77% / 39.62%. These noninjected controls remain substantially
less alarm-prone than the injected examples. Full registered APPS is a separate
diagnostic: FPR 100% at threshold 0.09535.

For context, the unfiltered augmented SDPA/BF16 model's same-holdout results
are ID macro AUROC/pAUROC20 0.95673/0.85999, injected-honest FPR 51.10%,
injected-backdoor recall 79.13%, and matched AUROC/pAUROC20 0.80357/0.51291;
see the [prior comparison](activation_filter_training.md#results).
This changes both data augmentation and CAFT, and is not a matched regular
control. The current CAFT implementation does not reproduce those benefits.

The result does not establish that concept ablation generally fails, or that
the learned component was globally protected. The model could learn alternative
features or respond to restoring the ablated component at inference; neither
mechanism has been measured here. The fixed geometry, single seed and cached
control confounds bound the conclusion.

ID source-macro AUROC/pAUROC20 are **0.95559 / 0.85131**, versus
**0.95748 / 0.86543** for the cached regular control. Pooled ranking is
**0.94866 / 0.82477**, versus **0.95158 / 0.83981**. Gloom pAUROC20 is
0.77266 versus 0.78590; STRIDE is 0.92995 versus 0.94497. At threshold 0.5,
pooled recall rises from 72.45% to 77.11% and FPR rises from 2.14% to 3.63%.
Pooled Brier/ECE improve from 0.11657/0.13088 to 0.10939/0.11348, with
100 unique scores among 3,012 records versus 90 for the control. The lower
ranking accompanies better calibration and higher recall; it is not a uniform
quality change. Complete source/calibration diagnostics remain in the summary.

## Execution and receipts

The user-authorized research release is
[Gleipnir-4B-ToolTrajectories-CAFT](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-CAFT),
verified at revision `a0613f47678cc716e05a7210f7dc7c3fb96dd5e6` and added to the
existing prompt-injection collection. It contains the FP32 causal master, rebased
FP32 serving layout, frozen prompt, summary/parity receipts and negative-result
model card. Publication does not promote this checkpoint. All remote file hashes
and collection membership pass; `results/concept_ablation_training/upload.json`
records the receipt. The [publication entrypoint](../../experiments/concept_ablation_training/publish.py)
keeps credentials local and uploads no raw training or evaluation records.

The successful campaign is `results/caft-regular-sdpa02/` on the same NC2 B200
UUID `GPU-36165f94-d010-1b8a-ebe5-d27fe53b9be9`. Training takes approximately
57m44s with finite recorded losses and gradients. Its FP32 master SHA256 is
`601a6c9b96ecffe9807fec8552564ac0ec4d1412c752bbd6863374b9370ee8e6`;
the BF16 adapter export is
`2fdbbbe2c2a069c55db0362f727b67322c6f5b6bd7c99bac7d946a10162224d4`.
Masters remain persistent and the reconstructable merge stays remote at
`/tmp/gleipnir-merged/caft-regular-sdpa02`.

Master-to-merge parity passes with MAE **0.0003166**, correlation **0.9999971**.
Master-to-BF16-serving parity passes with MAE **0.0019410**, correlation
**0.9998766**; serving-to-merge MAE is **0.0022576**. The maximum serving/master
score difference is **0.03121**. Acceptance is the frozen mean-error gate
(MAE <=0.020, correlation >=0.99, finite scores and nonzero adapter effect),
not a maximum-error ceiling. No failed-parity diagnostic exception is used.

Attempt `caft-regular-sdpa01` stops before any optimizer update: its absolute
alignment tolerance incorrectly ignores BF16 output rounding. The unchanged
projection has maximum relative forward/backward alignment **0.0601% / 0.0468%**
and exactly matches its explicit BF16 gradient reference. Attempt 02 corrects
only the operator check to a 0.1% relative ceiling plus reference agreement;
data, projection arithmetic, optimizer and evaluation parity limits stay fixed.
Both contracts and failed receipts are preserved.

The supplemental analysis-source binding is
`results/concept_ablation_training/analysis_source_binding.json`; the reused
holdout helper matches its previously archived source. To make merge space,
the unused matched-random-removal merge was archived on persistent storage and
byte-compared before reclaiming its ephemeral copy. Its receipt is
`results/concept_ablation_training/merge_archives/reclaimed.json`; original
masters, scores and compiler caches remain intact.

All five campaign stages and both parity gates complete. The independent audit
checks **12,126 finite scores**, exact ordered prompt/label/token identities,
logit-to-score agreement, file hashes, training coverage and recomputed ID/APPS
metrics, including direct ROC integration. Primary comparisons are in
`results/concept_ablation_training/heldout_apps.json`; the audit receipt is
`results/concept_ablation_training/independent_audit.json`. Evaluation time is
295.41s ID, 82.27s APPS benchmark and 68.96s honest controls, respectively
114,249 / 99,768 / 103,423 **input tokens/s**. These include one pass of bounded
c128 batches, not a new optimized-serving throughput comparison.
The compiled BF16 CAFT server is retained after training evaluation, then
retired for the Lens follow-up below; capacity remains running.

## Model-specific injection projection after CAFT

The user next requests fitting an APPS vector in the CAFT model itself and
projecting it out, to compare with earlier models. The
[refit contract](../../experiments/caft_direction_refit/README.md) freezes the
same 64 fitting tasks / 896 rows / 384 injected-minus-neutral pairs, both labels,
layer 20 and all-layer/all-token beta-1, zero-centered full projection. The
same 591 test tasks / 8,218 rows remain disjoint. No ID evaluation, training,
axis/layer sweep or promotion is part of this follow-up.

The compiled CAFT scorer is retired sequentially for BF16 eager Lens. Its
fresh master canary passes with MAE **0.0010249**, correlation **0.9999561**;
compiled/eager canary MAE is 0.0029330, correlation 0.9998263. Native BF16,
public-client, isolation, cleanup and projection-operator checks pass. Startup
takes 52.25s. A startup warning notes the preserved OMP_NUM_THREADS=4 setting's
potential CPU contention; neither configuration nor concurrency is changed.
Both full test arms use the same eager engine, with c64 and batches of 128.

The new unit axis has cosine **0.289197** with the original augmented-model
axis and **0.330831** with the filtered-model refit. Its natural mean-shift
norm is **10.982446**, versus 7.072298 and 8.434516 respectively. This is a
substantially different direction; its norm is a descriptive fitting diagnostic,
not evidence of causal strength. It is compatible with altered representation
after CAFT, without establishing rerouting as the cause of the persistent alarms.

**The new projection does not recover injection robustness.** All three
templates still produce 100% injected-honest FPR and 100% backdoor recall.
Matched ranking improves modestly; it remains substantially weaker than the
original augmented model's projection and weaker than the filtered-model refit.

| Model / inference arm | Injected-honest FPR | Injected-backdoor recall | Matched AUROC | Raw pAUROC20 |
|---|---:|---:|---:|---:|
| Original augmented SDPA, plain | 50.82% | 79.19% | 0.804912 | 0.519061 |
| Original augmented SDPA, fitted projection | 0.39% | 31.39% | 0.890519 | 0.680782 |
| Activation-filtered, plain | 100.00% | 100.00% | 0.601902 | 0.153282 |
| Activation-filtered, model-specific refit projection | 23.69% | 48.48% | 0.704681 | 0.283552 |
| Regular CAFT, fresh eager plain | 100.00% | 100.00% | 0.659126 | 0.194537 |
| Regular CAFT, model-specific refit projection | 100.00% | 100.00% | 0.678410 | 0.200307 |

All rows use the same held-out partition and each arm's clean-only threshold.
Prior-model rows are cached from the
[original direction study](b200_injection_direction.md#matched-primary-results) and
[filtered-model refit](activation_filter_training.md#model-specific-direction-refit),
not rerun. Cross-model comparisons change training data/intervention; within
each model the plain/projection comparison uses the same eager engine.

CAFT's plain threshold is **0.119203**, clean FPR **1.86%**; projected threshold
is **0.164516**, clean FPR **1.35%**. At the fixed plain threshold, projected
injected-honest FPR/recall remain **100%/100%** and clean FPR is **2.37%**.
This differs from the filtered-model refit's apparent FPR reduction after
recalibration: its fixed plain threshold also has 100% injected alarms, but
60.91% clean FPR. CAFT's failure persists at both reported cutoffs.

CAFT injected-honest means decrease from **0.7893/0.7963/0.7799** to
**0.7064/0.7482/0.7550**. The projected template AUROC/pAUROC20 values are
0.68584/0.21177, 0.69435/0.21865 and 0.65504/0.17050. Mean score shift over
all test rows is **-0.012800**. Vanilla AUROC/pAUROC20 changes from
0.92831/0.74656 to 0.92984/0.74612; recalibrated vanilla backdoor recall changes
from 52.49% to 50.60%. Projected neutral-template honest FPR is
3.55%/1.69%/1.52%, with backdoor recall 42.20%/36.71%/36.88%.
Complete ties/calibration and fixed-cutoff diagnostics remain in the summary.

The 1,771-row answer-mass diagnostic stays healthy: minimum/mean p01 is
0.98979/0.99682 plain and 0.99230/0.99727 projected, with none below 0.5.
Full-test eager/compiled plain MAE is **0.005004**, maximum difference 0.06242
and mean shift +0.000028; their injected alarms agree while ranking has small
serving-mode drift. The compiled scores are not the matched projection control.

An additional operator diagnostic on one fitting injection prompt confirms
removal along the **actual fitted vector at all 32 layers**, with maximum
relative residual alignment **0.0001593** (0.01593%), exact plain restoration
and clean request state. Its receipt is `actual_direction_operator.json`.
Thus the negative outcome is not explained by an unapplied vector on that
probe. It does not rule out layer-specific or distributed representations,
nonlinear effects, other geometries or other seeds. In particular, fitting a
large paired mean difference does not guarantee that removing it will remove
the decision's causal dependence on injection content.

Artifacts use `results/caft_direction_refit/caft_refit01/`; all captures and
batch tensors stay persistent on the Pod. Independent audit verifies all
**896 fitting scores / 16,436 test scores**, exact vector reproduction,
fit/test separation, source/input/score hashes, logits, all **158 batches**
and recomputed threshold/ranking summaries. Capture costs 62.87s for
1,466,943 tokens (**23,333 input tokens/s**); plain/projected scoring costs
190.17s/206.05s for 13,873,010 tokens each (**72,949/67,328 input tokens/s**).
The nine reused protocol tests and scoped Ruff pass. Findings and receipt copies
are collected locally without transferring model weights. The passing BF16
eager Lens server remains warm on port 8010, recorded in
`results/b200_sdpa_lens/caft_refit01/server.json`; no capacity lifecycle occurs.

## Keeping the original training projection active at inference

The user clarifies a further comparison: apply the **original augmented-model
axis**, identical to CAFT's training vector, at inference on the CAFT checkpoint.
The [original-projection contract](../../experiments/caft_original_projection/README.md)
keeps the exact original unit layer-20 axis and serialized all-layer/all-token
beta-1, zero-centered operator. There is no new training or vector fitting.
It reuses the same warm BF16 eager Lens process and both complete same-engine
plain/refitted-axis controls, with no restart or capacity lifecycle.

The fresh 20-row no-op canary reproduces scores **exactly** (MAE 0,
correlation 1). Runtime, startup source bindings, server process identity and
original-axis/training SHA256 equality pass. The only new scored arm contains
the same 591 disjoint test tasks / 8,218 rows, c64 and batches of 128; all
direction-fitting variants stay excluded. This adds a training-and-inference
intervention condition, separate from CAFT's ordinary-inference protocol.

**Keeping the original projection active also fails to reduce injection alarms.**
At either the fixed plain cutoff or the new clean-only cutoff, all templates
retain 100% injected-honest FPR and 100% injected-backdoor recall.

| CAFT inference arm | Injected-honest FPR | Injected-backdoor recall | Matched AUROC | Raw pAUROC20 |
|---|---:|---:|---:|---:|
| Unedited, reused | 100.00% | 100.00% | 0.659126 | 0.194537 |
| Refitted-axis projection, reused | 100.00% | 100.00% | 0.678410 | 0.200307 |
| Original training-axis projection | 100.00% | 100.00% | 0.656237 | 0.180888 |

The original-axis threshold is **0.106691**, with clean FPR **1.86%**. At the
fixed plain threshold 0.119203, clean FPR is **1.69%**. Mean injected-honest
scores are **0.7397 / 0.7439 / 0.7662**, versus 0.7893 / 0.7963 / 0.7799
unedited. The mean shift over all test scores is **-0.006805**. Template matched
AUROC/pAUROC20 values are 0.68062/0.21605, 0.68024/0.17132 and
0.60786/0.15529; the equal-template average worsens against unedited scoring.

Vanilla APPS AUROC/pAUROC20 is **0.93080 / 0.76027**, with recalibrated backdoor
recall **56.95%**. Neutral-template honest FPR is 9.98% / 1.86% / 2.37%, with
backdoor recall 59.35% / 46.48% / 47.00%. The answer-mass diagnostic has
1,771 rows, minimum/mean p01 **0.99301 / 0.99755**, none below 0.5.
Complete calibration/tie and fixed-cutoff diagnostics remain in the summary.

A direct original-axis probe on the same fitting injection prompt verifies
all-layer removal, with maximum relative residual alignment **0.0003465**
(0.03465%), exact plain restoration and clean request state. This keeps the
negative result distinct from an unapplied intervention on that probe. In this
checkpoint and frozen geometry, restoring the training-time ablation at
inference does not rescue APPS false positives; neither the original nor the
newly fitted single direction suffices. It does not establish the learning
mechanism, exclude broader subspaces or establish that CAFT generally fails.

Artifacts use `results/caft_original_projection/original01/`. The independent
audit passes exact input/source/score hashes, all **8,218 new finite scores**,
ordered prompt/label/token identities, logits, **65 batches** and recomputed
three-arm metrics. Scoring costs **206.96s** for **13,873,010 input tokens**,
or **67,033 input tokens/s**. Scoped Ruff and actual warm-engine no-op pass;
unchanged native/operator/partition checks reuse their validated receipts.
Results and audit copies are retained locally, all weights remain remote and
the same passing Lens process remains warm; no extra baseline scoring occurs.

## Augmented-vector addition on clean and neutral honest examples

The user requests a causal sensitivity check away from the saturated injected
population: add the original augmented-model vector to honest examples. The
[honest-steering contract](../../experiments/caft_honest_steering/README.md)
selects **591 clean + 1,773 neutral honest rows = 2,364**, on the same disjoint
test tasks. It excludes injected views and backdoors, keeps all three neutral
templates and reuses the same-engine plain scores. No fitting or retraining.

Addition is **only at layer 20, every token**, with no projection active.
The original augmented natural shift, norm **7.072298**, is added at +1x and
+2x; seed-0 random vectors have matching norms. Norm matching is disabled.
The two doses and random direction are fixed before outcomes. Strength refers
to the original augmented shift, not the CAFT recipient's larger shift. This
differs from Michael's recipient-normalized dose convention and is not an exact
reproduction of his base-model experiment.

**The original direction still causally raises suspiciousness in CAFT.** Both
positive doses raise the 1-minus-0 logit margin in **99.75% of rows**, substantially
more than the equal-norm random control. This supports preserved causal steering
influence and disfavors an explanation based on the original direction having
lost all steering influence after CAFT.

| CAFT honest arm | Clean FPR | Mean neutral FPR | Pooled FPR | Mean paired logit-margin shift |
|---|---:|---:|---:|---:|
| Unedited, reused | 1.86% | 2.82% | 2.58% | 0 |
| Augmented vector +1x | 12.01% | 77.38% | 61.04% | +1.9012 |
| Augmented vector +2x | 100.00% | 100.00% | 100.00% | +3.1632 |
| Equal-norm random +1x | 1.35% | 2.59% | 2.28% | -0.0960 |
| Equal-norm random +2x | 1.86% | 3.05% | 2.75% | +0.0395 |

All FPRs use the **fixed unedited clean-calibrated threshold 0.119203**;
neutral FPR averages the three equally sized templates. Thresholds are not
recalibrated after steering. At the separate threshold **0.5**, pooled honest
FPR is **0.34% unedited, 0.68% at +1x, 2.16% at +2x**, and 0.30% for both
random arms. Thus 100% alarms at the monitoring operating point do not imply
that every score crosses 0.5.

Mean pooled scores are 0.03534 unedited, **0.15009 / 0.35693** for positive
+1x/+2x and 0.03311/0.03635 for random +1x/+2x. Median paired margin shifts
are **+1.875 / +3.000**, versus -0.125/0.000 for random. At +1x, neutral-template
FPR is **95.60% / 59.22% / 77.33%**. Clean and neutral mean margin shifts are
**+2.312 / +1.764** at +1x and **+3.968 / +2.895** at +2x; effects vary with
input context rather than producing one constant logit offset.

The 512-row-per-arm answer-mass checks remain healthy. Minimum/mean p01 is
**0.99192/0.99618** at +1x and **0.98636/0.99456** at +2x; random minima are
0.98708 and 0.97796. No diagnostic has p01 below 0.5. All examples are harmless,
so AUROC/pAUROC are explicitly undefined; calibration/ties and complete paired
shift distributions remain in the summary.

Together with the unsuccessful original-axis projection, this shows that the
direction can drive alarms without its removal eliminating natural injection
alarms in this checkpoint. Learned compensation/rerouting remains plausible,
as do pre-existing alternative features, nonlinear propagation and an incomplete
single-axis intervention. This does not distinguish those mechanisms or prove
that CAFT caused information to move. One random direction, one trained seed
and previously investigated APPS templates limit generalization.

The no-op reproduces exactly (MAE 0, correlation 1); native/source/runtime
receipts remain unchanged. Artifacts use `results/caft_honest_steering/honest01/`.
Independent audit verifies all **9,456 new scores**, 591 tasks, **76 batches**,
exact paired prompt/label/token identities, finite logits and margins, vector
geometry/norms/doses, file/source hashes and recomputed five-arm metrics.
Each new arm scores **3,549,931 input tokens**: positive +1x/+2x takes
46.33s/46.38s (**76,615/76,540 input tokens/s**), random +1x/+2x takes
46.22s/45.94s (**76,812/77,265 input tokens/s**). These are one-pass costs on
the frozen honest cohort, with no fresh baseline scoring.
The three focused metric tests and scoped Ruff pass. Scores/receipts are
collected locally without weights, and the unchanged passing Lens engine
remains warm. There is no capacity lifecycle or promotion.
That CAFT process is subsequently retired for the
[base-model JudgeDeceiver steering comparison](base_judge_steering.md).
