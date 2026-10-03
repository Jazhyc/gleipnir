# Judge-targeted preference continuation

Date: 2026-10-03. Experiment:
[`experiments/judge_injection_continuation/README.md`](../../experiments/judge_injection_continuation/README.md).

The regular full-data 4B adapter avoids otherwise preferred answers carrying
judge-targeted injections. Continuing on balanced preference examples removes
most of this failure on held-out source queries. Transfer to harm monitoring is
partial: regular-prompt APPS injected-honest alarms fall from 100% to 70.64% at
clean-code 2% FPR operating points, with some loss in vanilla-backdoor detection.
ID source-macro pAUROC@20 falls from 0.8463 to 0.7958 and calibration worsens.
The model is more conservative, but the experiment does not produce an overall
monitoring improvement. The completed campaign receives no model promotion.

## Frozen intervention and data

Continue a copy of the regular Qwen3.5-4B rank-128 FP32 master LoRA. The original
adapter remains unchanged. One epoch, fresh AdamW, LR 2e-5, 3% warmup, linear
decay, logical batch 32, full BF16 base weights and FP32 adapters. Reuse the
validated packed FlashQLA/SDPA B200 recipe and shared persistent compiler cache;
retain finite/missing-gradient guards. All 524 updates completed. Training
runtime was 508.3 seconds including approximately 256 seconds for the first
compiled update; steady updates took approximately 0.47 seconds. The initial
trainable-tensor digest matches the original adapter's final digest.

Pool the full JudgeDeceiver release at
`ShiJiawenwen/JudgeDeceiver@7cfc96af0272befd83220b1d7d3b0699d072ecbf`, including
its original attack-training and evaluation populations and case studies.
All 2,790 source rows are accounted for. Deduplication yields 1,260 preference
pairs, retaining merged provenance and applicable released suffixes. Main/RLAIF
labels invert the attacker's desired answer after matching the bad candidate;
search preferences compare each other entry with the attack target without
ranking the other entries. Tool selection uses its explicit `real_label`.
These are construction labels, separate from attacker-desired targets.

Keep both A/B orders for each pair. Include clean, preferred-answer-injected and
disfavored-answer-injected variants, applying each applicable released suffix
separately. Preferences remain fixed when suffixes are attached. Use a shared
quality-judging prompt with native nonthinking chat and a direct A/B boundary.
Hard two-class cross-entropy selects the existing A/B token logits; no new
classification head, teacher annotations or monitoring replay are introduced.

Split by normalized original query before augmentation. All answer pairs,
orders and suffixes for a query stay together. There are 25 training and 6
held-out query groups. The realized split is 16,760 / 4,188 augmented examples
(80.008% / 19.992%), representing 1,008 / 252 unique pairs. Inputs plus one
decision token total 5,964,368 / 1,525,718 tokens. No truncation is required.
Tool selection has one query and remains entirely in training.

The final checkpoint is fixed in advance. No held-out preference or APPS result
selects an epoch, learning rate, prompt, threshold or checkpoint. APPS is excluded
from training and data generation. The user narrowed monitoring evaluation to
the regular prompt before held-out scoring; no aggressive/conservative full
evaluation is performed. The interrupted sweep remains in `evaluation/`.
Active results are under `results/judge_injection_continuation/evaluation_regular/`.

## Held-out preference comparison

Both checkpoints receive exactly the same held-out task prompt and A/B inputs.

| Population | Examples | Original accuracy | Continued accuracy | Original mean p(correct) | Continued mean p(correct) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 504 | 99.40% | 99.80% | 0.912841 | 0.997156 |
| Preferred answer injected | 1,842 | 58.36% | 100.00% | 0.557874 | 0.999987 |
| Disfavored answer injected | 1,842 | 95.98% | 99.46% | 0.862530 | 0.994446 |
| Pooled | 4,188 | 79.85% | 99.74% | 0.734588 | 0.997209 |

For preferred-answer injections, the original clean-to-injected correct-to-wrong
flip rate is 41.21%; after continuation it is 0%. Mean paired p(correct) change
improves from -0.371998 to +0.000765. Disfavored-answer injections give -0.067342
and -0.004776 respectively. This asymmetry supports injection-driven avoidance
by the original checkpoint within the preference task.

Pooled AUROC rises from 0.911884 to 0.999979, Brier improves from 0.123010 to
0.002553, and log loss from 0.370631 to 0.017864. Query-macro accuracy rises
from 82.22% to 98.39%; source-macro accuracy rises from 83.58% to 97.59%.
The continued model makes no errors on held-out LLMBar, MT-Bench or RLAIF
examples. All 11 remaining errors are search examples: one clean and ten with
the disfavored answer injected. Search accuracy is 90.35% versus 88.60% before
continuation; its labels have the weaker construction-based provenance.
There are 2,758 / 1,416 repeated pooled scores and 147 / 0 exact-half ties.

## Regular-prompt APPS transfer

One monitor prompt, the original regular 0/1 rubric. The three rows below are
fixed attack templates, not a sweep of monitor instructions. Use 655 clean
honest examples to set each checkpoint's strict score > threshold rule,
yielding 13/655 clean alarms (1.985%). Original threshold: 0.164516; continued:
0.075858. Compare 647 paired honest/backdoor problems per template.

| Attack template | Original injected-honest alarms | Continued injected-honest alarms | Continued neutral-honest alarms | Original matched injected AUROC | Continued matched injected AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-4 | 100.00% | 79.85% | 2.14% | 0.649131 | 0.723193 |
| GPT-4.1 | 100.00% | 99.85% | 1.37% | 0.609265 | 0.677687 |
| GPT-4o | 100.00% | 32.21% | 1.53% | 0.673455 | 0.788816 |

Equal-template mean injected-honest alarms fall from 100% to 70.64%; neutral
honest alarms are 1.48% / 1.68%. Mean injected-honest score falls from 0.637405
to 0.108354, and the mean injection-minus-neutral shift from 0.593938 to
0.097343. Matched injected AUROC rises from 0.643951 to 0.729899; matched
neutral AUROC changes from 0.896688 to 0.901123. These gains do not eliminate
false alarms, particularly for the GPT-4.1 template.

Vanilla-backdoor AUROC falls from 0.918908 to 0.907030, and TPR at clean 2% FPR
from 51.78% to 45.44%. Continued injected-backdoor TPR is 93.97%, 100% and
74.50% by template, versus 100% previously. At the separate fixed 0.5 boundary,
continued injected-honest alarms are 0%, but only 2.74% of all benchmark
backdoor variants exceed that boundary. Lower scores alone do not establish
improved discrimination or calibration.

## Regular-prompt ID sanity check

Use the same 3,012 CoT-removed ID inputs as the original adapter, including
visible assistant prose. Input SHA256 is
`8e0fde58119098356feb0d897194d0dddec5e21b3e5e6959d5495d0999bb0ed2`.
There is one regular-prompt ID pass, without a monitor-instruction sweep.

| Weights | Source-macro pAUROC@20 | Source-macro AUROC | Pooled recall at 0.5 | Pooled FPR at 0.5 | Pooled Brier | Pooled ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original regular | 0.846273 | 0.951394 | 81.90% | 6.41% | 0.097827 | 0.058943 |
| Continued | 0.795786 | 0.932206 | 62.13% | 1.78% | 0.171919 | 0.190329 |

The macro pAUROC@20 decrease is 0.050486. Gloom decreases from 0.763290 to
0.675476; STRIDE from 0.929255 to 0.916097. Their continued AUROCs are 0.883620
and 0.980793 respectively. Pooled AUROC decreases from 0.942615 to 0.922744,
and pooled pAUROC@20 from 0.816665 to 0.760267. Pooled log loss worsens from
0.318370 to 0.586217. Mean predicted probability falls from 0.474922 to
0.343536 against prevalence 0.533865. There are 2,102 distinct ID scores and
910 repeated scores, versus 1,569 and 1,443 previously. Both ranking and
calibration regress, particularly on Gloom; this is more than a uniform
probability-scale change.

## Verification and limits

Six master-to-serving checks pass across base/original/continued weights and
preference/regular-monitor cohorts. Minimum correlation exceeds 0.999; maximum
mean absolute score difference is 0.007978. Both adapters have a nonzero effect
under both surfaces. Reference reuse is explicitly recorded and checksum-checked
after narrowing the evaluation scope. One persistent vLLM engine supplies raw
requested decision-token logprobs and constrained one-token responses.

All 20,502 fresh preference, APPS and ID outputs are collected and locally checked for exact row
membership, labels/metadata, rendered prompt hashes, finite raw logprobs and
score-formula agreement. FP32 master and serving adapter checksums match their
training receipt. The original-ID integrity receipt initially existed only
locally; the failed startup was preserved, the verified receipt transferred,
and evaluation resumed without retraining.

The final summary SHA256 is
`621b681b1b78d40dd11fc7eb02d4f9c67cb561ac9d98e80eb4aaa6df5e517620`.
The continued FP32 master and serving weights are respectively
`636246da8cf3d00cfe5ca1b9c2cded5d3392935404494dd854315986d0cd5399` and
`5f793247fe37f2de5df1faeb31f68b980394be3dcb166367e365039cd81da0cc`.
Local proofs are saved as `local_preference_verified.json`,
`local_apps_verified.json`, `local_id_verified.json`, and the combined
`local_coverage_verified.json` alongside the final summary.

The first final-report attempt failed because the legacy ID metric helper
expects per-example token counts while the scoring run recorded totals. Retain
that failure. Reporting now reuses the original ID token counts only after
matching every rendered prompt hash, ID, label and source, and matching the
actual new scoring total (33,750,959) and maximum (29,513) tokens. No scores or
prediction files are altered, and no GPU rescoring is performed. A checksum-pinned
reporting-only compatibility path retains the old scoring identity; inference
still rejects source drift. Preserve the scoring source snapshot and record the
reporting source separately. Future scoring records per-row token counts.

The 4,188 held-out variants cover only six source queries, not thousands of
independent tasks. Suffixes were optimized against other judge models and
upstream output formats; this is not an adaptive attack evaluation against
the Qwen A/B interface. Construction labels, dominant main-task populations,
the all-training tool query and a single seed limit generalization. Preference
quality and harmfulness are distinct targets, so strong auxiliary-task results
must be interpreted alongside APPS and ID rather than treated as monitor labels.

The auxiliary objective teaches the model to retain preference judgments despite
judge-targeted text. Harmfulness false alarms improve for some APPS templates,
but remain high and accompany worse ID detection/calibration. This one-epoch
continuation with no monitoring replay does not isolate which data or objective
mechanism causes the tradeoff. It supports further work on matched benign
injection exposure under the monitoring label contract, rather than replacing
the regular monitor with this continued checkpoint.
