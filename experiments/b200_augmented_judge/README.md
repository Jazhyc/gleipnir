# Same augmented adapter, BF16 and optimized JudgeDeceiver judging

User requested the original A/B JudgeDeceiver result on both serving backends
for the completed BF16 MLP/FA4-trained adapter used in the ID/APPS comparison.
Hypothesis: the optimized recipe changes injection avoidance in quality judging.
Keep the same FP32 adapter and existing BF16 merge; no training or merging.

Reuse all 4,188 frozen original variants (504 clean, 1,842 preferred-injected,
1,842 disfavored-injected), both orders, 252 pairs and six held-out query groups.
Preserve the original prompt and labels. This adapter did not train on this data.
Historical regular/augmented scores are qualified context, not same-adapter
controls. Source provenance and unresolved license follow the
[original contract](../judge_injection_continuation/README.md).

Compare compiled BF16 vLLM 0.31 with the selected FP8 attention-projection,
FP4 MLP/GDN and MXFP8 recipe. Keep native tokenizer, causal LAST cached head,
synchronous corrected scheduler, prefix caching off, 32K envelope, 128-row
partitions and c128. Select A/B embedding rows 32/33 instead of monitor rows
15/16, with exact merged-weight and native-kernel audits. The optimized worker
uses a process-local A/B token binding in the existing head auditor; its original
source and all backbone kernel checks remain checksum-bound. The BF16 worker
supports an explicit subclass decision-row selection, defaulting to 0/1.

Before held-out scoring, require finite FP32-master/merged/backend agreement
on the original 30 training-derived A/B canaries (MAE <=0.020, correlation >=0.99,
nonzero adapter effect >1e-6). Stop on a failed gate, drift, truncation, OOM,
nonfinite or incomplete scores. Preserve all failures; no inherited waiver.
Freeze inputs, commands and executed sources before replacing the recorded
monitor process. Reuse persistent caches and retire each server before the next.

Report accuracy with the original p(B)>=0.5 tie rule, pooled/source/query ranking
and calibration, score ties, paired clean/injection shifts and flips, and paired
backend changes. Report input tokens/s, requests/s and latency; one pass per
backend is descriptive timing. No threshold tuning, checkpoint/recipe selection
or promotion. Preference labels do not define harmfulness FPR. Six queries and
dependent suffix/order variants limit generalization.

Run `python -m experiments.b200_augmented_judge.run --stage prepare` locally,
then `--stage all` in the selected serving runtime on the existing NC2 B200.
The driver first runs bounded references in the preserved training runtime,
then BF16 and optimized scoring. Keep the final passing optimized A/B scorer
warm. No capacity changes. Active-turn monitoring only: no heartbeat tool is
available. Results/data/logs use `b200-augmented-judge02`. The first attempt stopped before reference scoring
because its subprocess omitted the existing reference dependency overlays; its
failed log/status and parent retirement receipt remain under `judge01`.
The retry uses the shared `training_environment` helper, with no package changes.

## Authorized diagnostic follow-up

BF16 completed all 4,188 variants. Optimized A/B parity failed before holdout
scoring (master MAE 0.071835, correlation 0.959254; merged MAE 0.067990).
Outputs, adapter effect and native dispatch pass. The user explicitly authorizes
finishing optimized scoring as a failed-parity diagnostic.
`b200-augmented-judge-diagnostic01` reuses the frozen `judge02` BF16 predictions
and references, preserving its failed gate and retired-worker receipt.
The changed scope does not relabel numerical failure as parity.
Run `--stage prepare`, then `--stage diagnostic`; repeat finite/effect/native
checks after restart, retain diagnostic scope even if fresh parity passes, and
keep the finite diagnostic optimized A/B scorer warm.
