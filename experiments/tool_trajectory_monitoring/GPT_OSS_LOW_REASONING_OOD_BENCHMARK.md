# GPT-OSS-120B low-reasoning strict-OOD benchmark

This frozen evaluation tests whether actual model-native reasoning explains the
weak immediate-logit GPT-OSS-120B result. It uses all 6,395 rows in the same
paper-author-filtered six-source OOD suite and the identical full binary teacher
prompt.

The model receives `Reasoning: low` through its pinned Harmony template. It
samples with OpenAI's recommended temperature 1 and top-p 1 settings until the
first Harmony `<|end|>` closes the native analysis channel. The evaluator then
preserves those exact token IDs, appends a canonical final-channel
`Prediction:` boundary, and constrains the next token to literal `0` or `1`
while retaining both raw logprobs. This measures uncertainty after reasoning
without conditioning on a sampled final decision.

The intervention is compared pairwise with the completed immediate-final-logit
artifact. No OOD label may select a reasoning trace, prompt, request setting,
threshold, retry rule, subset, or stopping point. A 12-row source-label canary
must begin in the Harmony analysis channel, terminate at its first end token,
avoid the 16,384-token cap, and return both decision-token logprobs. Any full-run
format failure stops the condition rather than excluding a row.

Run on one RTX Pro 6000:

```bash
sbatch cluster/slurm/benchmark_gpt_oss_low_reasoning_ood.sh
```

Generations, predictions, and summaries are written under the ignored directory
`results/tool_trajectory_monitoring/gpt_oss_120b_low_reasoning_ood/`. Re-running
the same command resumes checksum-validated reasoning traces and predictions.
