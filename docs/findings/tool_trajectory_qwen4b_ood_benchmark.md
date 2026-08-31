# Qwen3.5-4B full-teacher-prompt OOD benchmark

Date: 2026-08-30. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This benchmark measures the unadapted
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` model on the
six paper-author OOD action-only datasets. Qwen receives Gleipnir's full binary
teacher prompt, including the detailed scheming specification. The pinned
non-thinking chat template ends with an empty `<think>...</think>` block, after
which the evaluator appends `Prediction:` and measures the normalized relative
logit of the literal `1` versus `0` tokens. No rationale is sampled.

The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
the input SHA-256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`,
and the frozen config SHA-256 is
`ceee751409281aef3f6a9e2d0e9f6f2a733e64741cd71e807d4a6136539e82ee`.
The 4B tokenizer vocabulary, merges, configuration, and chat template are
byte-identical to the pinned 9B benchmark. The OOD suite was never used to tune
the prompt, score, request settings, or threshold.

## Paper-comparable result

The paper's primary aggregate is Mean-OOD pAUROC@20: the unweighted arithmetic
mean of the six independently computed per-dataset pAUROC@20 values.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.8549** | **0.6175** |
| Pooled examples | 6,395 | 0.7973 | 0.4718 |
| AgentDojo | 1,500 | 0.8359 | 0.4528 |
| CUA-SHADE-Arena | 955 | 0.8627 | 0.7006 |
| IaC | 1,443 | 0.8826 | 0.6302 |
| MLE-Sabotage | 148 | 0.9600 | 0.9061 |
| Rogue Eval | 1,499 | 0.8198 | 0.6089 |
| SHADE-Arena | 850 | 0.7682 | 0.4064 |

At the fixed 0.5 threshold, source-macro balanced accuracy is `0.7029`, macro
Brier score is `0.21552`, macro recall is `0.4766`, and macro FPR is `0.0707`.
The pooled balanced accuracy is `0.6880` and pooled Brier score is `0.21708`.
There are 1,726 unique continuous scores; 4,669 rows share a score with at
least one other row.

MLE-Sabotage is the strongest source, while SHADE-Arena and AgentDojo are the
largest low-FPR weaknesses. The 4B result is `0.0801` below the matched 9B
base's Mean-OOD pAUROC@20. It nevertheless becomes the lowest-cost observed
frontier point and strictly dominates the GPT-OSS-120B base result, which is
both more expensive and less accurate.

## Exact cost accounting

The exact rendered Qwen contexts contain 60,539,619 input tokens, or
9,466.7113 per evaluation. The maximum is 30,382 tokens, below the frozen
32,768-token engine limit. The selected-token interface produces exactly one
output token per row.

OpenRouter's public model catalog did not list Qwen3.5-4B on 2026-08-30, so
there is no current OpenRouter route whose price can be attached to this row.
For paper comparability, the canonical coordinate uses the paper's explicit
Artificial Analysis hosted proxy of `$0.03/M` input tokens and `$0.15/M`
output tokens:

```text
input  = 9,466.7113 * $0.03 / 1000 = $0.284001 per 1K evaluations
output =       1.0 * $0.15 / 1000 = $0.000150 per 1K evaluations
total                                  $0.284151 per 1K evaluations
```

The frontier therefore records **`$0.2842/1K`**. This is a dated hosted proxy,
not an OpenRouter quote or the cost of the local Slurm run. It assumes no
prompt-cache or batch discount.

## Runtime and artifact audit

Slurm job `30975535` completed in 26 minutes 44 seconds on one NVIDIA RTX PRO
6000 Blackwell Server Edition using vLLM 0.24.0, PyTorch 2.11.0+cu130, BF16
weights, one CPU, and 32 GB host RAM. The evaluator's measured interval after
engine construction was 1,464.67 seconds. The persistent text-only engine used
continuous batching, a one-token constrained response, and explicit decision-
token logprobs. All 6,395 rows are present exactly once; there were no missing
logprobs, truncations, invalid decisions, duplicate IDs, or backend failures.

The ignored result artifacts are:

- `results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/predictions.jsonl`,
  SHA-256 `9981f99d77c5e342c6e590386df8396fb5d2b5393bcb87119ac0cb3e67df141c`;
- `results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/canary_result.json`,
  SHA-256 `74439851c437d58d08b034b0d360c1710cb38ac71a79808a3f81b65e8f3cf030`;
- `results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/result.json`,
  SHA-256 `274204270a0a269f942ce16078ba4f6705fdaed26e5b89edc339f3acd4aa867f`.

The canonical cost--performance comparison is maintained in
[`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md).

- [Qwen3.5-4B model and revision history](https://huggingface.co/Qwen/Qwen3.5-4B)
- [OpenRouter model catalog search for Qwen3.5-4B](https://openrouter.ai/models?q=qwen3.5-4b)
- [Sinha et al., Appendix D.4 metric definition](https://arxiv.org/html/2605.29601v1#A4.SS4)
- [Sinha et al., Appendix E hosted proxy prices and cost accounting](https://arxiv.org/html/2605.29601v1#A5.SS4)
- [`tool_trajectory_qwen_ood_benchmark.md`](tool_trajectory_qwen_ood_benchmark.md)
