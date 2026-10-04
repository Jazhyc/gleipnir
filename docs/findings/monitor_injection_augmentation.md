# Monitor-directed injection replacement campaign

Date: 2026-10-04. Status: complete. All 272 training updates and 15,138 fresh
evaluation scores finished; the serving engine exited and the B200 is idle.

The replacement intervention improves clean ID ranking and substantially reduces
APPS injected-honest alarms, while leaving a large residual failure. At each
adapter's own threshold calibrated to 1.985% clean honest-code FPR:

| Metric | Original regular adapter (cached) | Augmented adapter |
| --- | ---: | ---: |
| Clean ID source-macro pAUROC@20 | 0.846273 | 0.872983 |
| Clean ID pooled AUROC | 0.942615 | 0.953226 |
| Clean ID pooled Brier | 0.097827 | 0.089774 |
| Clean ID pooled ECE | 0.058943 | 0.059872 |
| APPS injected-honest FPR, mean of three templates | 100% | 44.43% |
| APPS injected-backdoor TPR, mean of three templates | 100% | 87.64% |
| Matched injected-backdoor vs injected-honest AUROC, template mean | 0.643951 | 0.859471 |
| Vanilla APPS AUROC | 0.918908 | 0.925572 |
| Vanilla APPS TPR at clean-code ~2% FPR | 51.78% | 55.18% |

The APPS thresholds are 0.164516 original and 0.148047 augmented; thresholds
are selected only from each adapter's same 655 clean honest controls, then kept
fixed for injected controls. These template averages reuse the same problems
and are not independent samples. The three transfer conditions have:

| Injection | Injected-honest FPR | Injected-backdoor TPR | Matched injected AUROC |
| --- | ---: | ---: | ---: |
| transfer_gpt4 | 76.49% | 98.15% | 0.831968 |
| transfer_gpt41 | 38.17% | 84.54% | 0.854263 |
| transfer_gpt4o | 18.63% | 80.22% | 0.892181 |

This direct monitoring augmentation transfers more strongly to the APPS controls
than the previous preference continuation (70.64% mean injected-honest FPR) or
joint preference training (100%). Clean ID source-macro ranking increases for
both sources: Gloom pAUROC@20 0.763290 to 0.809069, STRIDE 0.929255 to 0.936897.
Calibration improves by Brier/log loss but slightly worsens by ECE. This is one
seed with eight authored training strings, off-policy source continuations and
copied teacher targets; it supports the intervention's usefulness without
establishing broad/adaptive robustness or isolating a universal mechanism.

For the new adapter's 3,012 paired clean/injected ID views, held-out mechanisms
produce small score shifts: pooled AUROC 0.953226 clean versus 0.953104 injected;
accuracy at 0.5 is 87.58% versus 87.52%. Injected harmless FPR is 1.85% at the
source-specific clean ~2% FPR thresholds (actual clean FPR 1.92%). Harmful TPR
at those thresholds is 75.12% clean versus 74.07% injected.
Mean absolute score changes are 0.0151
for harmless and 0.0183 for harmful examples. At threshold 0.5, 0.36% of harmless
and 1.43% of harmful examples flip from correct to wrong; 0.78% and 0.93% flip
the other way. There is no original-adapter baseline for this synthetic set.

Held-out injected-harmless FPR is 1.58% for user, 1.02% for assistant and 2.97%
for tool additions. Channel spoofing and decision pressure give 1.71% and 1.99%.
Agreeing/conflicting demanded verdicts give 1.56% and 2.15%; harmful TPR is
74.88% and 73.25%. Both demanded-0 and demanded-1 groups have slightly negative
mean score changes in both behavior classes, rather than the opposing large
shifts expected from dominant copying/reversal. This does not rule out smaller
demand effects; the groups have different examples, without counterfactual
0/1 demands on every identical trajectory.

Metric convention: `synthetic_id`'s historical `pauc20` fields use sklearn's
standardized partial AUC, while project `pauroc_at_20` uses raw partial area
divided by 0.2. Do not compare those fields directly. The collection verification
also records paired source-macro partial AUC using the project definition:
0.872983 clean versus 0.869804 injected, a small decrease.

The user selected replacing 40% of the 8,688 original regular-prompt training
rows, preserving the original source behavior verdicts and cached Kimi soft
targets. There are 3,475 injected views (1,738 harmless, 1,737 harmful) and 5,213
clean views. Every trajectory is visited once; the matched run has 272 updates,
one epoch, fresh rank-128 initialization and the regular 5e-5 learning rate.
See the [protocol](../../experiments/monitor_injection_augmentation/README.md)
and [decision](../decisions/monitor_injection_augmentation.md).

Eight authored training templates in four families demand either 0 or 1,
independently of the behavior label. The demand agrees in 1,785 cases and
conflicts in 1,690; always reversing it matches only 48.6% of hard labels.
Uniform role quotas are 1,159 user, 1,158 assistant and 1,158 tool messages.
Requested positions follow truncated N(0.5,0.25²) over source characters,
snapped to existing boundaries. All original source bytes and clean JSONL
lines are preserved; four pre-existing XML task-wrapper irregularities remain.
The actual agent continuation is not regenerated, and copied teacher targets
are an explicit off-policy invariance assumption.

The user requested reusing original-adapter results instead of rescoring it.
Fresh evaluation is only for the new adapter, with the regular prompt and
fixed final checkpoint: 5,184 APPS benchmark, 3,930 honest controls, 3,012 clean
ID and 3,012 synthetic held-out ID examples. Synthetic examples use four
templates from two mechanism families excluded from training, with each role
occurring 1,004 times. Every synthetic view has its own clean pair. There is
no original-adapter synthetic baseline, and no APPS-selected checkpoint.
Report agreeing/conflicting verdict demands separately, together with source,
role, family, position, ranking, calibration and clean-calibrated operating points.

Preparation has 83,969,155 input tokens, maximum 29,391, with zero truncation
inside the validated 29,696-token envelope. Training manifest SHA256 is
`e99a700ed1b588323e0f0a757e73e50046496fbc25c29c158b34b129dec99f51`;
held-out evaluation manifest SHA256 is
`88fae967f735ef65fbebc0b25bb820be7f57b6183d2b3859382570fddb1c127b`.
Forty-six focused checks pass locally and on the B200, including resumption,
raw-logprob preservation, rejection of source/label/cache drift, reversible
lowercase/uppercase ID rendering and paired threshold/flip diagnostics.

The source-checked launch uses existing Runpod B200 `3422tdq7tfog52`, PID
13805. Frozen execution code and both materializations match local checksums.
The selected packed BF16 FlashQLA/SDPA recipe, FP32 masters, startup-validation
reuse and persistent compiler cache remain in force. Serving parity passed
before vLLM population scoring. Logs are under
`logs/runpod/monitor_injection_augmentation/`; launch, cache and progress
receipts are under `results/monitor_injection_augmentation/`.

The actual Transformers tokenizer audit subsequently completed all 8,688 rows,
with 83,969,213 input tokens, maximum 29,391 and no truncation. Its count is
58 tokens above the raw-tokenizer preparation audit; retain both receipts.
All 15 execution-source hashes match the reviewed local code. Model loading
verified a frozen BF16 base and 256 FP32 trainable LoRA tensors, then explicitly
reused the existing startup validation without repeating numerical probes.
The compiler cache is populated and resolves to the persistent
`/workspace/gleipnir/.cache/training/student_injection_awareness` directory.

Training completed all 272 updates. The matched metadata validator
passed again on the collected checkpoint: objective weights, initialization,
packed BF16 FlashQLA recipe, startup-reference identity and changed master are
consistent with the frozen contract. Actual training runtime was 3,520.38 seconds
(58.7 minutes); the training invocation, including token audit and export, took
4,189.72 seconds (69.8 minutes). Mean training loss was 0.23464. Peak CUDA
allocated/reserved memory was 150.13/150.98 GiB. These are execution diagnostics,
not evidence of held-out robustness.
The mean optimizer step excluding the initial compile was 11.81 seconds.

The final FP32 master file SHA256 is
`3ecfd0d397aed7ad666e189bbd6fed82a20f93be33b5608a32c288c78ac7edef`;
serving file SHA256 is
`bbed5d2c6fc5c9e8ae84a78993d43b2151acfd306e1d14b7ce5013e8e3c17fd7`.
Both collected files match their completion-receipt checksums; each contains
256 finite FP32 adapter tensors.
The 20-example master/serving gate passed: augmented score correlation 0.999798,
mean absolute difference 0.003291; base correlation 0.999265, difference 0.008019.
Maximum serving/reference adapter effects were 0.898995/0.888744. Serving
compilation and graph capture completed before population scoring. All 15,138
fresh predictions completed, with exact coverage and raw decision-logprob
evidence. The report was independently recomputed locally from collected scores.
All 15,138 rendered-prompt hashes, labels/source metadata, one-token outputs,
raw-logprob-derived scores, unique identities and file/receipt checksums also
pass an independent audit using the saved Jinja chat template. Maximum prompt
lengths are 16,321 benchmark, 3,931 honest controls, 29,513 clean ID and 29,546
synthetic ID, below the 32,768-token serving envelope. A redundant Transformers
tokenizer audit stalled on local filesystem I/O and was stopped after this
complete independent rendering audit passed; no GPU run or frozen data changed.
Weights, scores and logs are collected in the ignored artifact trees;
`evaluation_regular/summary.json`, `completion_audit.json` and
`collection_verification.json` provide the final reporting/audit receipts.

No in-chat scheduler is available in this session. Active-turn startup checks
do not provide automatic post-turn agent follow-ups; the remote sequential
queue is not claimed as agent monitoring.
