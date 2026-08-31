# Qwen3.5-9B full-teacher-prompt OOD benchmark

Status: completed on all 6,395 rows on 2026-08-30.

This frozen evaluation measures the unadapted `Qwen/Qwen3.5-9B` base on all
6,395 rows in the paper-author-filtered six-source OOD suite. The hypothesis is
that the base model provides useful continuous action-monitoring scores under
the full instructed teacher prompt, not only under the compact student prompt.

The intervention is the existing direct selected-token logit interface. The
exact teacher prompt is wrapped with the pinned Qwen3.5 non-thinking chat
template, which ends the assistant prefix with an empty `<think>...</think>`
block. The context then ends at `Prediction:`. vLLM constrains the next token to
literal `0` or `1`; both raw logprobs and their normalized positive probability
are retained. There is no sampled reasoning or visible rationale.

The frozen Kimi K3 evaluation with the same teacher prompt and decision-token
semantics is the principal prompted-monitor baseline. GPT-OSS-120B is a second
model-family comparison using its model-native final-channel boundary. The
completed Qwen3.5-9B compact-prompt ID result is context only, not a matched OOD
baseline. No prompt, setting, subset, threshold, or stopping decision may be
selected from OOD labels. Report every source, the unweighted source macro, and
the pooled population for AUROC and the paper's raw normalized pAUROC@20, plus
calibration, threshold, and tie diagnostics.

The full token audit covers every row: the longest Qwen input is 30,382 tokens
and the suite contains 60,539,619 prompt tokens under a 32,768-token engine
limit. The run first scores one row from every source-label stratum (12 rows)
as an interface-only canary, then resumes the remaining rows in the same
persistent vLLM engine. Stop on any artifact, prompt, model-revision,
tokenization, chat-boundary, logprob, truncation, or row-membership drift
declared in `qwen_ood_benchmark.json`.

Run on one RTX Pro 6000:

```bash
sbatch cluster/slurm/benchmark_qwen_ood.sh
```

Predictions and summaries are written under the ignored directory
`results/tool_trajectory_monitoring/qwen35_9b_teacher_ood/`. Re-running the
same command resumes from checksum-validated predictions.

The completed run reached `0.6976` Mean-OOD pAUROC@20 and `0.8850` Mean-OOD
AUROC. Its exact uncached OpenRouter list-price projection is `$0.9468/1K`
evaluations. Full per-source metrics, accounting, runtime, and artifact hashes
are recorded in
[`../../docs/findings/tool_trajectory_qwen_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_ood_benchmark.md).
