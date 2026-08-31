# Qwen3.5-9B visible-reasoning binary-logit OOD benchmark

Status: completed 2026-08-31.

## Hypothesis and intervention

This matched ablation tests whether one visible deliberation improves the
unadapted model's binary uncertainty estimate. The primary endpoint is raw
normalized Mean-OOD pAUROC@20: compute pAUROC over `FPR <= 0.20` independently
for each of the six frozen OOD sources, divide by 0.20, and take their
unweighted mean. Standard Mean-OOD AUROC is secondary.

The model remains the same `Qwen/Qwen3.5-9B` BF16 revision used by the completed
immediate-logit baseline. Native Qwen thinking stays disabled. The intervention
changes the full binary monitor instruction from silent review and immediate
`Prediction:` to a visible concise analysis ending at `Prediction:`. At that
exact boundary, the evaluator requests both literal `0` and `1` logprobs and
normalizes them with the same two-class softmax as the immediate baseline.

Reasoning is greedy and capped at the paper-aligned 4,096-token reserve. If a
rationale reaches the cap, or otherwise completes without the exact terminal
prefix, the evaluator retains the generated analysis, appends `Prediction:`,
records the recovery, and extracts the binary logits without regenerating the
trace. Empty outputs and unsupported finish reasons fail closed.

## Baseline, selection rule, and stop condition

The paired baseline is the completed immediate binary-logit run at `0.6976`
Mean-OOD pAUROC@20 and `0.8850` Mean-OOD AUROC. Both arms use the same 6,395 row
IDs, labels, model revision, full binary semantics, non-thinking assistant
boundary, literal decision tokens, threshold, and temperature-zero terminal
decision. The intervention adds visible analysis and changes the instruction's
output procedure, so it is an interface-level test-time-compute ablation.

OOD labels must not select the prompt, settings, threshold, subset, boundary
recovery, or stopping point. Report all sources and the paired comparison
regardless of direction. Stop on checksum or context drift, invalid token
logprobs, incomplete canary coverage, unsupported rationale termination, or
incomplete paired coverage.

Paper-comparable hosted accounting counts the initial rendered prompt once and
the rationale, terminal `Prediction:` prefix, and one decision token as output.
The second vLLM continuation is an implementation detail, not a second logical
hosted request.

The audited initial contexts total 60,584,384 tokens, averaging 9,473.7113 per
row, with a minimum of 1,040 and maximum of 30,389. The maximum plus the frozen
4,096-token rationale reserve and one decision token is 34,486, below the
36,864-token engine limit.

## Result

The definitive run reached `0.6923` Mean-OOD pAUROC@20 and `0.8607` Mean-OOD
AUROC. The paired immediate binary-logit baseline reached `0.6976` and `0.8850`,
so visible reasoning changed the metrics by `-0.0054` and `-0.0243`,
respectively. It cost `$1.0054/1K` under the dated hosted-price convention,
6.2% more than direct logits.

The direction is heterogeneous: AgentDojo pAUROC@20 improves by `0.2980`, but
all five other sources decline. Fifteen rows use the frozen boundary recovery
and none are dropped. The complete per-source, score-behavior, cost, and run
audit is in
[`../../docs/findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md).

## Reproduction

Run the balanced 12-row canary first:

```bash
sbatch cluster/slurm/benchmark_qwen_binary_reasoning_ood.sh \
  --stop-after-canary
```

Then resume the same checksum-bound cache for the full run:

```bash
sbatch cluster/slurm/benchmark_qwen_binary_reasoning_ood.sh
```

Artifacts are written under the ignored directory
`results/tool_trajectory_monitoring/qwen35_9b_teacher_binary_reasoning_ood/`.
