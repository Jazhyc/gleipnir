# GPT-OSS-120B full-teacher-prompt OOD benchmark

Date: 2026-08-30. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This benchmark measures the unadapted
`openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a` base on the
six paper-author OOD action-only datasets. The model receives Gleipnir's full
binary teacher prompt, including the detailed scheming specification. The
pinned Harmony context enters the final channel and ends at `Prediction:`;
vLLM constrains the next token to literal `0` or `1` and retains both selected
token logprobs. The score is the normalized probability of `1` within that
two-token decision boundary. No rationale is sampled.

The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
the input SHA-256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`,
and the frozen config SHA-256 is
`9b8d613595e56e6ed55ebaf50149c3321b9918b69307f534676e69c42c24a838`.
The OOD labels never selected the prompt, score, model revision, request
settings, threshold, subset, retry rule, or stopping point.

## Paper-comparable result

The primary aggregate is the paper's raw normalized Mean-OOD pAUROC@20: the
unweighted arithmetic mean of the six independently computed per-dataset
values.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.7862** | **0.4789** |
| Pooled examples | 6,395 | 0.7281 | 0.3424 |
| AgentDojo | 1,500 | 0.8746 | 0.6400 |
| CUA-SHADE-Arena | 955 | 0.8247 | 0.5684 |
| IaC | 1,443 | 0.7253 | 0.2443 |
| MLE-Sabotage | 148 | 0.9527 | 0.8944 |
| Rogue Eval | 1,499 | 0.6204 | 0.1952 |
| SHADE-Arena | 850 | 0.7193 | 0.3311 |

At the fixed 0.5 threshold, source-macro balanced accuracy is `0.6462`, macro
Brier score is `0.33049`, macro recall is `0.3653`, and macro FPR is `0.0730`.
The pooled balanced accuracy is `0.6291` and pooled Brier score is `0.32686`.
There are only 734 unique continuous scores across 6,395 rows, so ties are
material and the retained raw logprobs should be used for any later audit.

## Exact cost accounting

The exact rendered Harmony contexts contain 53,597,573 input tokens, or
8,381.1686 per evaluation. The maximum is 27,473 tokens, below the frozen
32,768-token engine limit. The selected-token interface produces exactly one
output token per row.

GPT-OSS is open-weight and has no first-party hosted API price. The paper used
OpenRouter's `$0.039/M` input and `$0.18/M` output hosted-token proxy for its
GPT-OSS-120B rows. Applying that same proxy preserves comparability with the
paper rows in the canonical frontier:

```text
input  = 8,381.1686 * $0.039 / 1000 = $0.326866 per 1K evaluations
output =        1.0 * $0.18  / 1000 = $0.000180 per 1K evaluations
total                                      $0.327046 per 1K evaluations
```

The canonical row therefore records **`$0.3270/1K`**. OpenRouter prices vary
by route and over time, so this is a hosted proxy rather than a local Slurm
cost or a timeless model property. Using the 2026-08-30 headline/lowest listed
`$0.03/$0.17` route gives `$0.2516/1K`; using a `$0.15/$0.60` higher-priced
route gives `$1.2578/1K`. No prompt-cache or batch discount is assumed.

## Runtime and artifact audit

Slurm job `30972405` completed in 48 minutes 8 seconds on one NVIDIA RTX PRO
6000 Blackwell Server Edition using vLLM 0.24.0, PyTorch 2.11.0+cu130, the
upstream MXFP4 weights with Marlin, four CPUs, and 64 GB host RAM. The
persistent text-only engine used continuous batching, a one-token constrained
response, and explicitly requested decision-token logprobs.

All 6,395 frozen input IDs are present exactly once. Source counts and labels
match the manifest; there are no missing logprobs, non-numeric scores,
truncations, duplicate IDs, invalid decisions, or backend failures. The ignored
result artifacts are:

- `results/tool_trajectory_monitoring/gpt_oss_120b_ood/predictions.jsonl`,
  SHA-256 `96b9bba16d0371d1ec0ae8c91dd595e9daee486a57b5e252906238cdd3cec830`;
- `results/tool_trajectory_monitoring/gpt_oss_120b_ood/canary_result.json`,
  SHA-256 `aaa10d4d25bae463314d4ccbd8cfdd42178c93523db20499f69d539147181661`;
- `results/tool_trajectory_monitoring/gpt_oss_120b_ood/result.json`, SHA-256
  `569e05968980d8ea71dc1f42754a12ecde01cea697c5d9e1919ace703263bd08`.

## Interpretation

The unadapted base is strong on the small MLE-Sabotage split and moderately
useful on AgentDojo and CUA-SHADE-Arena, but its low-FPR ranking is poor on IaC,
Rogue Eval, and SHADE-Arena. The `0.4789` Mean-OOD pAUROC@20 is far below both
Kimi K3's matched-interface `0.9084` and the paper's GPT-OSS-120B SFT result of
`0.718`.

At the paper-matched proxy price, this result is technically a frontier point
only because its one-token output makes it about `$0.020/1K` cheaper than the
next-lowest paper row. It is a weak endpoint, not an attractive quality point:
GPT-OSS-20B SFT costs only slightly more and improves pAUROC@20 by `0.1771`.
At the higher `$0.15/$0.60` route, the base result is dominated and leaves the
frontier. The base and SFT interfaces also differ, so their gap is descriptive
rather than an isolated causal estimate of fine-tuning.

The canonical comparison is maintained in
[`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md).
