# Qwen3.5-27B full-teacher-prompt OOD benchmark

Status: frozen and awaiting evaluation on all 6,395 rows.

This evaluation measures the unadapted `Qwen/Qwen3.5-27B` model on every row
in the paper-author-filtered six-source OOD suite. The hypothesis is that dense
Qwen3.5 capacity scaling from 4B through 9B to 27B improves low-FPR action-
monitoring rankings under the full instructed teacher prompt.

The intervention changes only the Qwen3.5 backbone capacity. The exact frozen
OOD artifact, teacher prompt, non-thinking chat boundary, `Prediction:` prefix,
literal `0|1` selected-token logprobs, and metric implementation match the 4B
and 9B runs. No rationale is sampled.

The completed Qwen3.5-4B and 9B benchmarks are the primary within-family
baselines; Kimi K3 is cross-family context. No prompt, setting, subset,
threshold, or stopping decision may be selected from OOD labels. Report every
source, the unweighted source macro, and the pooled population for AUROC and raw
normalized pAUROC@20, plus calibration, threshold, and tie diagnostics.

The pinned 27B tokenizer and chat template are byte-identical to the smaller
Qwen checkpoints. The suite therefore contains the same 60,539,619 prompt
tokens (9,466.7113 per row), and its longest input is 30,382 tokens under the
32,768-token engine limit. Literal `0` and `1` are each one token. The run first
scores one row from every source-label stratum (12 rows) as an interface-only
canary, then resumes the remaining rows in the same persistent vLLM engine.

Run on one RTX Pro 6000:

```bash
sbatch cluster/slurm/benchmark_qwen27b_ood.sh
```

Predictions and summaries are written under the ignored directory
`results/tool_trajectory_monitoring/qwen35_27b_teacher_ood/`. Re-running the
same command resumes from checksum-validated predictions.
