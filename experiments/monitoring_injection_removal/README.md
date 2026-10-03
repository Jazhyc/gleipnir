# Standard-prompt 4B with aggressive census filtering

User requested 2026-10-03 after the behavior-grounded prompt failed to reduce
injected-honest alarms. The user explicitly selected removing **all 1,154 rows
with the frozen unadapted Qwen3.5-9B census score >=0.5**, including exact ties.
This is a model-flag filter with known false positives and false negatives;
do not describe the resulting corpus as verified injection-free.

Hypothesis: removing the broad injection-candidate subset changes the trained
monitor's injection-text sensitivity. Intervention: train a fresh standard-prompt
Qwen3.5-4B on the remaining 7,534 rows (4,330 source-label-0, 3,204 source-label-1).
Keep the original order, trajectory identities and individual Kimi K3 target
records for retained examples. Use the **original regular instruction**, frozen
in `student_prompt.txt`; no behavior-grounded training prompt, new teacher calls,
counterexample generation or new labeling is part of this experiment.

Reuse the selected packed BF16 B200 recipe, pinned backbone and zero-initialized
rank-128/alpha-256 FP32 adapters, seed 0, AdamW 5e-5, soft BCE only, logical batch
32, 16,384-token physical packs, intact long singletons, FlashQLA/SDPA, no model
checkpointing and the persistent shared compiler cache. Keep one epoch; this
means **236 updates**, not 272. Reuse recorded unchanged-recipe startup validation
and mark diagnostic probes not performed; retain provenance and finite/missing
gradient guards. Adapter-specific serving parity remains mandatory.

The census input, prediction, contract and prompt/model identities are pinned in
`config.yaml`. Every census row must match the source trajectory, label, lineage
and original student-prompt hash. Scores must be finite and agree with their raw
decision logprobs. Reject incomplete, duplicate, foreign or stale census records.
The selected ID exclusion list, before/after source-label counts, filtered data
hashes and unchanged retained teacher provenance are materialized under
`data/monitoring_injection_removal/`. The original full training set and all
previous artifacts are preserved.

Baseline: the earlier regular-trained 4B is the direct comparison using the same
training instruction. Also retain the completed base, aggressive-trained and
behavior-grounded models' frozen three-prompt injection grids (twelve existing
cells). Check all existing prediction contracts, completions and hashes before
reuse; record these as cached results, not rescoring. Add the new filtered adapter
under neutral, aggressive and conservative monitor prompts, giving **15 grid
cells**. Use the exact prior APPS membership, payloads and honest-code controls
from `mkenney2/monitor-injection-evals` branch `feat/gleipnir-vllm-grid`, with the
same source revision and checksums. Score 5,184 original rows and 3,930 honest
controls per prompt. The mapping is unchanged: neutral = original regular,
aggressive = earlier injection-aware, conservative = approved behavior-grounded.

The ID sanity check runs once on all 3,012 frozen CoT-removed examples with
the new model's **neutral / standard training prompt**. The ID set has no prompt
sweep. Use frozen trajectory/lineage holdouts. Select only the final one-epoch
checkpoint; there is no evaluation-driven checkpoint, threshold or prompt tuning.

For serving, score the immutable twenty-example mechanical parity cohorts with
the original-FLA causal master and unadapted base under all three prompts. The
cohorts come from the preceding campaign and may include training-excluded rows;
they are forward-only implementation checks, not training data or model selection.
Then use one persistent BF16 vLLM engine, continuous batching, LoRA, one constrained
decision token, the native nonthinking `Prediction:` boundary and both requested
raw decision logprobs. Require correlation >=0.99, MAE <=0.02 and a nonzero adapter
effect before any full evaluation. Score ID first, then the three injection cells.
This is 30,354 fresh predictions and 109,368 verified reused predictions.

Report injected-honest alarms at the original clean-honest threshold, matched
injected/neutral backdoor AUROC, original per-template AUROC and TPR@2%FPR, score
shifts/ties and ID source-macro ranking, calibration and fixed-threshold metrics.
Stop on source/cache identity drift, nonfinite/missing gradients or scores, failed
serving parity, truncation, OOM or incomplete coverage; preserve failure receipts.

Interpretation limits: this removes 1,140 harmful-labeled and 14 benign-labeled
examples, changes source composition and reduces training exposure/updates.
An effect would establish a difference from this broad removal intervention,
not isolate prompt-injection content as the cause. No effect would not rule out
other data mechanisms or classifier misses. Some reviewed STRIDE and Gloom prompt
exposures remain below the user-selected threshold. APPS labels target code
backdoors, while training and ID cover broader agent behavior. One seed and fixed
transfer templates do not establish adaptive robustness. No winner promotion.

Use the existing authorized B200 Pod `3422tdq7tfog52`; no new allocation is needed.
Artifacts live in `data/monitoring_injection_removal/`,
`results/monitoring_injection_removal/` and
`logs/runpod/monitoring_injection_removal/`. This session has no in-chat agent
scheduler: startup checks occur in the active turn, with no automatic post-turn
heartbeat promised. The remote evaluation queue serializes compute only.

```bash
python -m experiments.monitoring_injection_removal.prepare
python -m experiments.monitoring_injection_removal.train
python -m experiments.monitoring_injection_removal.launch_evaluate
```
