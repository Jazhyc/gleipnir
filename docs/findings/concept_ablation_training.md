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
The BF16 CAFT server remains warm, recorded in
`results/caft-regular-sdpa02/server.json`; capacity remains running.
