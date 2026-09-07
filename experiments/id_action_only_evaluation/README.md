# Gleipnir 4B cleaned-ID evaluation

Hypothesis: removing assistant prose and explicit thinking calls changes the
standard Gleipnir 4B checkpoint's ID discrimination and calibration, especially
on STRIDE where the original prompts include reasoning. Evaluate all 3,012
cleaned ID rows exactly once; do not select a checkpoint, prompt, threshold or
subset from the resulting scores. OOD is excluded.

Intervention: the separately hashed output of `id_assistant_prose_audit`, with
all assistant prose and explicit `think` calls removed. Baseline: the completed
standard mixed-data rank-128 Gleipnir 4B evaluation on the original ID prompts,
`results/tool_trajectory_monitoring/gleipnir4b_id_v1/`. This is the same
`soft-n21837-mixed-qwen35-4b-seed0` checkpoint, not a later training ablation.
Preserve compact instructions, model revision, FlashInfer backend, non-thinking
chat boundary, binary decision logprobs, threshold 0.5 and source-macro metrics.
This bundles visible-prose and reasoning removal; it cannot isolate their effects.
The historical baseline uses a separate inference invocation, so tiny score
changes may include BF16/batching variation.

## Frozen execution

The user requested the two GPUs on the existing `gleipnir-improvement` target.
Probe on 2026-09-07: Ubuntu 24.04, driver 580.126.20, two NVIDIA H100 80GB HBM3
GPUs with 81,559 MiB each; both idle with no compute processes. Use one TP1
persistent vLLM engine per GPU, continuous batching and one decision token.

First compare four source/label-balanced cleaned rows between FP32 causal-master
(eager, GPU 0) and BF16 serving adapter (vLLM, GPU 1), including the unadapted
base to confirm nonzero adapter effect. Require the established correlation
>=0.99, mean absolute score difference <=0.02 and adapter effect >=1e-6.
Pinned FLA/causal-conv kernels must pass preflight before eager model import.

Then alternate intact 128-row batches across the two GPUs using the existing
sharding runner: 1,536 and 1,476 rows. Keep a frozen empty resume snapshot and
separate per-GPU output paths. Require exact disjoint coverage, matching config
hashes and finite scores at merge. Report pooled and per-source/macro AUROC,
pAUROC@20, Brier score, threshold diagnostics and ties plus paired score changes.

Stop on parity failure; input, checkpoint or prompt drift; missing logits;
context overflow; GPU errors; incomplete coverage; or exhausted bounded evaluator
hang recovery. Do not overwrite prior predictions. The process watchdog is not
an agent heartbeat. This session has no scheduling tool; active-turn checks
cover startup and completion, without promising a wake-up after the turn ends.

## Reproduce

```bash
.venv/bin/python -m experiments.id_action_only_evaluation.prepare
# Push this experiment, prepared input/manifest and result-root jobs/status using
# scripts/lambda_cloud.py; run on the already reserved Lambda target:
.venv/bin/python -u -m experiments.id_action_only_evaluation.run_lambda
# After collecting results locally:
.venv/bin/python -m experiments.id_action_only_evaluation.summarize
```

Artifacts: `results/id_action_only_evaluation/`; runtime logs:
`logs/lambda/id_action_only_evaluation/` plus the shared sharding runner's
per-GPU logs under the result root. Original ID inputs and results remain frozen.
