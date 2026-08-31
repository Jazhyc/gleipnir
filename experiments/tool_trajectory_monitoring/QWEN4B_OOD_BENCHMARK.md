# Qwen3.5-4B full-teacher-prompt OOD benchmark

Status: completed on all 6,395 rows on 2026-08-30.

This evaluation measures the unadapted `Qwen/Qwen3.5-4B` model on every row
in the paper-author-filtered six-source OOD suite. The hypothesis is that the
smaller base model provides useful continuous action-monitoring scores under
the full instructed teacher prompt.

The intervention is the same direct selected-token logit interface used for
Qwen3.5-9B. The exact teacher prompt is wrapped with the pinned Qwen3.5
non-thinking chat template, ending at `Prediction:` after the empty assistant
`<think>...</think>` block. vLLM constrains the next token to literal `0` or
`1`; both raw logprobs and their normalized positive probability are retained.
There is no sampled reasoning or visible rationale.

The completed Qwen3.5-9B benchmark is the primary capacity baseline. Kimi K3
and GPT-OSS-120B provide matched full-prompt model-family comparisons. No
prompt, setting, subset, threshold, or stopping decision may be selected from
OOD labels. Report every source, the unweighted source macro, and the pooled
population for AUROC and raw normalized pAUROC@20, plus calibration, threshold,
and tie diagnostics.

The full token audit covers every row under the pinned 4B tokenizer: the suite
contains 60,539,619 prompt tokens (9,466.7113 per row), and its longest input is
30,382 tokens under the 32,768-token engine limit. Literal `0` and `1` are each
one token. The tokenizer vocabulary, merges, configuration, and chat-template
hashes are byte-identical to the pinned 9B benchmark. The run first scores one
row from every source-label stratum (12 rows) as an interface-only canary, then
resumes the remaining rows in the same persistent vLLM engine. Stop on any
artifact, prompt, model-revision, tokenization, chat-boundary, logprob,
truncation, or row-membership drift declared in `qwen4b_ood_benchmark.json`.

Run on one RTX Pro 6000:

```bash
sbatch cluster/slurm/benchmark_qwen4b_ood.sh
```

Predictions and summaries are written under the ignored directory
`results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/`. Re-running the
same command resumes from checksum-validated predictions.

The completed run reached `0.6175` Mean-OOD pAUROC@20 and `0.8549` Mean-OOD
AUROC. Its exact one-token hosted-price projection is `$0.2842/1K`
evaluations using the paper's Qwen3.5-4B proxy rate. OpenRouter did not list a
Qwen3.5-4B route on the evaluation date. Full per-source metrics, accounting,
runtime, and artifact hashes are recorded in
[`../../docs/findings/tool_trajectory_qwen4b_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen4b_ood_benchmark.md).
