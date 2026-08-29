# Qwen3.5 reasoning-mode source canary

This frozen pilot asks whether reasoning before the literal `0|1` decision
boundary improves the unadapted `Qwen/Qwen3.5-9B` monitor's ranking or
calibration. It is not the final in-distribution evaluation: the paper's exact
946-row STRIDE test and 2,066-row Gloom-Exfiltration manifests are not local.
The available research-use-only source release instead supplies two earlier
1,000-row synthetic families. The pilot reuses the already-frozen 100-row
teacher-canary selection, balanced across source and label and filtered to at
most 28,000 source tokens.

The three conditions are frozen in `qwen_reasoning_benchmark.json`:

1. `direct`: disable native thinking and score the immediate next-token logits
   after `Prediction:` using the compact student prompt.
2. `visible_reasoning`: disable native thinking, request a concise two-sided
   visible analysis, remove its generated terminal decision digit, and rescore
   literal `0|1` logits at that exact boundary.
3. `thinking_reasoning`: use the identical reasoning prompt and sampling
   settings with Qwen's native thinking chat template enabled, then apply the
   same terminal rescoring procedure.

Both reasoning conditions use the Qwen-recommended general reasoning sampling
family and allow up to 16,384 generated tokens. One persistent text-only vLLM
engine runs all arms. Outputs checkpoint atomically after every eight rows and
include the generated trace, parse status, terminal score, token counts, prompt
hashes, per-source metrics, and uncertainty diagnostics.

Materialize the ignored input from the pinned source files and teacher-canary
selection:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_qwen_reasoning_benchmark
```

Run a four-row backend canary, then resume the complete frozen pilot:

```bash
python -m experiments.tool_trajectory_monitoring.benchmark_qwen_reasoning --limit 4
python -m experiments.tool_trajectory_monitoring.benchmark_qwen_reasoning
```

Stop rather than interpreting the run if the condition row IDs, model revision,
prompt semantics, or reasoning sampling settings drift; if either decision token
is absent at the terminal boundary; or if any condition remains incomplete.
