# APPS black-box injection detection

Hypothesis: explicitly asking for prompt-injection presence lets the base
Qwen3.5-4B and current augmented SDPA replay detect payloads independently of
whether submitted code contains a backdoor. Compare the fixed pinned base
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and final 272-update
`b200-augmented-sdpa-replay02` checkpoint; no training or checkpoint selection.

Reuse the unchanged [census rubric](../training_injection_census/prompt.txt),
native non-thinking `Prediction:` boundary and cached causal LAST 0/1 scorer.
Use ordinary unquantized BF16 vLLM 0.31 serving for both models, sequentially
on the user-reserved NC2 B200. This is a user-requested exception to optimized
low-precision serving. Prefix caching is off; context/chunk budget is 32K,
concurrency and original batch partitions are 128. Preserve persistent caches.
Record current hardware/driver, model/file/source/prompt hashes, raw logits,
token counts, timing and complete coverage. Verify preserved merge/master
evidence, model checksums, fresh native BF16 audit and original 20-row
master-to-serving canary before the full evaluation of either model.

Score all 5,184 original APPS rows and 3,930 honest controls from the frozen
revision `58e722bf9912903d56a334f873e069aa17ccb7ad`. Change only the outer
instruction. Construction labels are 1 for `inj_transfer_*` and 0 for clean,
vanilla and `neutral_transfer_*`; preserve original behavior labels separately.
These labels describe the constructed monitor-targeted payload, not independent
annotation of every possible incidental instruction. No submitted code runs.

The prompt, models and populations are fixed before scoring. APPS has prior
research exposure and is a descriptive benchmark, not fresh model selection.
Report pooled and behavior-stratified AUROC/raw pAUROC20, per-template matched
injection-versus-neutral metrics, calibration, ties, paired shifts and strict
0.5 diagnostics. The empirical clean-only <=2% FPR threshold is descriptive,
with no held-out calibration claim. Task bootstrap retains all related views
and pairs model scores; three fixed templates do not establish broad robustness.

Stop on identity drift, failed parity/native audit, missing/nonfinite or truncated
scores, OOM, incomplete coverage or completion. Preserve failed and partial
receipts. Resume batches only with identical input/model/runtime/source identity.
No capacity changes. Startup/progress monitoring is active-turn only.

Run `python -m experiments.apps_injection_detection.run --stage prepare`, then
`--stage evaluate` through the existing `experiments.b200_vllm031.runtime`
wrapper. Outputs: `results/apps_injection_detection/presence01/`; logs:
`logs/runpod/apps_injection_detection/`. Collect artifacts locally on completion.
