# Matched Qwen/Kimi full-trajectory agreement

Hypothesis: confidence-scale and/or ranking differences between the local Qwen
prefix teacher and Kimi full-endpoint teacher may introduce conflicting student
supervision. Compare the same full trajectories and original detailed teacher
rubric; do not confound teacher identity with the visible trajectory length.

Freeze 64 rows per source/label cell (640 total across five training sources),
ordered by SHA256 of seed and ID, unique trajectory hashes within the sample.
Selection never reads teacher scores. Match every Kimi target by exact ID,
dataset, label and rendered-prompt SHA; retain original raw decision logprobs.
No new Kimi calls, ID/OOD tuning, target rescaling or student training.

Reuse the proven Qwen3.5-27B-FP8 TP2 non-thinking teacher evaluator on both H100s,
original Torch2.11/vLLM0.24 environment, prefix caching disabled. Singleton
scoring after the existing repeated-reference canary minimizes batching changes.
Retain the pinned model revision, raw decision logprobs, one-token interface,
prompt hashes and resumable records. Stop on provenance/context drift, invalid
scores, failed canary or GPU errors. Do not interrupt unrelated GPU work.

Report pooled and per-source probability MAE, signed Qwen-minus-Kimi bias,
confidence, binary-decision disagreement, Spearman rank agreement, and descriptive
log-odds slope/intercept. Raw logprobs across vocabularies are not directly
comparable; normalize over the same binary outcomes and compare log-odds margins.
Label-based Brier scores are descriptive on the sampled training population,
not held-out model-selection evidence. Source/label balance changes population
weights, and numerical FP8/interface differences remain part of this comparison.
Differences cannot by themselves establish why prefix student training failed.
Also stratify by the original Kimi provider because the cache spans three
providers with small, previously measured numerical differences.

Entrypoint: `python -m experiments.teacher_agreement.run prepare|run|analyze`.
Preparation consumes `config.yaml`, freezes prompt/Kimi/config artifacts under
`results/teacher_agreement/`, and audits exact Qwen token lengths. Run reuses the
shared evaluator and then computes agreement; never calibrates targets in place.

## Execution

2026-09-06: frozen sample passes the shared evaluator's input/config validation:
640 rows, 7,029,373 Qwen prompt tokens, maximum 30,149. Both reserved H100s
were idle before launch. Lambda supervisor PID 167521 uses the original `.venv`;
log: `logs/lambda/teacher_agreement/run.log`. Three focused statistics tests and
Ruff checks pass. Final numerical canary and agreement results remain pending.

Completed the same day: all 640 unique IDs passed local re-analysis, including
exact source-prompt hashes, labels, dataset identities, frozen config SHA and
raw-logprob-to-probability consistency. Numerical canary passed with mean
absolute probability difference 0.002924 and maximum 0.014350. Final results:
`docs/findings/matched_qwen_kimi_teacher_agreement.md`. Both teacher inputs and
Qwen outputs are retained locally under the ignored result directory.
