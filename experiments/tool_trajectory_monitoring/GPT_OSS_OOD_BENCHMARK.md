# GPT-OSS-120B strict-OOD benchmark

This frozen evaluation measures `openai/gpt-oss-120b` on all 6,395 rows in the
paper-author-filtered six-source OOD suite. The hypothesis is that the unadapted
model provides useful continuous action-monitoring scores under the full
instructed teacher prompt.

The intervention is a direct selected-token readout. Each exact teacher prompt
is wrapped with the pinned GPT-OSS Harmony template at low reasoning effort,
the assistant is placed directly in its final channel, and the context ends at
`Prediction:`. vLLM constrains the next token to literal `0` or `1`; both raw
logprobs and their normalized positive probability are retained. This is a
direct-logit benchmark, not a sampled-reasoning evaluation.

The frozen Kimi K3 evaluation with the same teacher prompt and decision-token
interface is the principal descriptive baseline. The companion Qwen3.5-9B OOD
condition uses the same full teacher prompt and its model-native direct-logit
boundary; compact-prompt Qwen results cover the ID sources only and are context.
No prompt, setting, subset, threshold, or stopping decision may be selected from
the OOD labels. Report every source, the unweighted source macro, and the pooled
population for AUROC and the paper's raw normalized pAUROC@20, plus calibration,
threshold, and tie diagnostics.

The token audit covers every row: the longest GPT-OSS input is 27,473 tokens
under a 32,768-token engine limit. The run first scores one row from every
source-label stratum (12 rows total) as an interface-only canary, then resumes
the remaining rows in the same persistent vLLM engine. Stop on any artifact,
prompt, model-revision, tokenization, logprob, truncation, or row-membership
drift declared in `gpt_oss_ood_benchmark.json`.

Run on one RTX Pro 6000:

```bash
sbatch cluster/slurm/benchmark_gpt_oss_ood.sh
```

Predictions and summaries are written under the ignored directory
`results/tool_trajectory_monitoring/gpt_oss_120b_ood/`. Re-running the same
command resumes from checksum-validated predictions.
