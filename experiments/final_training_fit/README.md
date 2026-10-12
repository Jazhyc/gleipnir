# Final-checkpoint teacher fit on the full training population

Hypothesis: the final matched regular 9B checkpoint fits Kimi K3 better than
4B on their shared training population. Measure all 8,688 original unfiltered
rows, their original regular student prompts and fixed Kimi probabilities.
Reuse final one-epoch seed-0 adapters from student_injection_awareness; no
training, filtering, held-out evaluation or model selection. Report row-pooled
and source-macro CE, teacher entropy, teacher-to-student Bernoulli KL and score
MAE, plus source/label breakdowns. Compute losses stably from raw logit margins.
These in-sample diagnostics do not establish capacity ceilings or OOD transfer.

Use the existing authorized NC2 B200, persistent base/master/export artifacts
and compiler caches. Score each adapter-specific merge sequentially in the
same ephemeral slot; preserve prior merge manifests and all original adapters.
BF16 backbone and FP32 two-row head is an explicit matched diagnostic exception
to the 4B-only optimized serving default. 9B has the same 32-layer attention/GDN
layout but a 4,096-wide head; audit width from its pinned text config. Preserve
all existing BF16 projection/native-attention checks. Require independently
saved FP32-master canary identity, master/merge and master/serving parity and
nonzero adapter effects for each size before full scoring. No parity waivers.

Freeze inputs and sources before launch. Score full rendered prompts without
truncation, 128-row groups/concurrency, context 32,768. Stop on drift, missing
or nonfinite scores, failed parity, foreign GPU/server ownership, or a two-hour
stage timeout. Preserve failed receipts; no automatic retries. Retire the owned
scorer at completion/failure, keep capacity running. The detached driver queues
both models automatically; no in-chat heartbeat tool is available.

The first attempt stopped before merging/scoring because original completion
receipts lack modern master/export SHA fields. Attempt 02 verifies the saved
parity master SHA, writes normalized receipts beside read-only adapter links,
and reuses the unchanged complete 01 workloads. Preserve the first failure and
original receipts. No checkpoint, objective or data changes.

Run remotely with the pinned training runtime:
`python -m experiments.final_training_fit.run --prepare`, then detached
`python -m experiments.final_training_fit.run`. The driver uses the training
runtime for merging/reference and vLLM 0.31 runtime for full scoring. Artifacts
live under `data/final-training-fit02-{4b,9b}` and corresponding `results/`;
queue status is `results/final-training-fit02/status.json`.
