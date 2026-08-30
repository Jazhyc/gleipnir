# Qwen3.5-9B full-teacher-prompt OOD benchmark

Date: 2026-08-30. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This benchmark measures the unadapted
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` base on the six
paper-author OOD action-only datasets. Qwen receives Gleipnir's full binary
teacher prompt, including the detailed scheming specification. The pinned
non-thinking chat template ends with an empty `<think>...</think>` block, after
which the evaluator appends `Prediction:` and measures the normalized relative
logit of the literal `1` versus `0` tokens. No rationale is sampled.

The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
the input SHA-256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`,
and the frozen config SHA-256 is
`d560d4174315833f3ad6d1737ca7f9c0d76e415e2ce66b4aa71df13d29de510d`.
The OOD suite was never used to tune the prompt, score, request settings, or
threshold.

## Paper-comparable result

The paper's primary aggregate is Mean-OOD pAUROC@20: the unweighted arithmetic
mean of the six independently computed per-dataset pAUROC@20 values.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.8850** | **0.6976** |
| Pooled examples | 6,395 | 0.8430 | 0.4968 |
| AgentDojo | 1,500 | 0.8792 | 0.5657 |
| CUA-SHADE-Arena | 955 | 0.8981 | 0.7814 |
| IaC | 1,443 | 0.9257 | 0.7348 |
| MLE-Sabotage | 148 | 0.9824 | 0.9602 |
| Rogue Eval | 1,499 | 0.9075 | 0.7225 |
| SHADE-Arena | 850 | 0.7171 | 0.4212 |

At the fixed 0.5 threshold, source-macro balanced accuracy is `0.7220`, macro
Brier score is `0.19063`, macro recall is `0.5394`, and macro FPR is `0.0954`.
The pooled balanced accuracy is `0.7055` and pooled Brier score is `0.19554`.
There are 1,587 unique continuous scores; 4,808 rows share a score with at
least one other row.

The largest weakness is SHADE-Arena, followed by AgentDojo in the low-FPR
region. MLE-Sabotage is the strongest source. The resulting `0.6976` Mean-OOD
pAUROC@20 is useful as an unadapted full-prompt baseline, but is not a new
cost--performance frontier point: lower-cost paper rows already achieve higher
pAUROC@20.

## Exact cost accounting

The exact rendered Qwen contexts contain 60,539,619 input tokens, or
9,466.7113 per evaluation. The maximum is 30,382 tokens, below the frozen
32,768-token engine limit. The selected-token interface produces exactly one
output token per row.

On 2026-08-30, the official
[OpenRouter Qwen3.5-9B page](https://openrouter.ai/qwen/qwen3.5-9b) listed
`$0.10/M` input tokens and `$0.15/M` output tokens. Applying those uncached list
prices gives:

```text
input  = 9,466.7113 * $0.10 / 1000 = $0.946671 per 1K evaluations
output =       1.0 * $0.15 / 1000 = $0.000150 per 1K evaluations
total                                  $0.946821 per 1K evaluations
```

The canonical frontier row therefore records **`$0.9468/1K`**. This is a hosted
list-price projection for comparison with the paper, not the cost of the local
Slurm run. It assumes no prompt-cache or batch discount.

## Runtime and artifact audit

Slurm job `30972555` completed in 40 minutes 43 seconds on one NVIDIA RTX PRO
6000 Blackwell Server Edition using vLLM 0.24.0, PyTorch 2.11.0+cu130, BF16
weights, one CPU, and 32 GB host RAM. The persistent text-only engine used
continuous batching, a one-token constrained response, and explicit decision-
token logprobs. All 6,395 rows are present exactly once; there were no missing
logprobs, truncations, invalid decisions, duplicate IDs, or backend failures.

The ignored result artifacts are:

- `results/tool_trajectory_monitoring/qwen35_9b_teacher_ood/predictions.jsonl`,
  SHA-256 `97e84749de2977cc566ee8823e515b944ead8f10f7997b08f1f93d7dc3ea87a5`;
- `results/tool_trajectory_monitoring/qwen35_9b_teacher_ood/canary_result.json`,
  SHA-256 `4a157654313021a641a4904fb9ca92a41cfb7843958e7768e8181f306d3aaeae`;
- `results/tool_trajectory_monitoring/qwen35_9b_teacher_ood/result.json`,
  SHA-256 `66152d21b4bf927135f22d3bbfa74e3ec53ee55cd6b344ccf25bcc9b92085016`.

The canonical cost--performance comparison is maintained in
[`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md).
