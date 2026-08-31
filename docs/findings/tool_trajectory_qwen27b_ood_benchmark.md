# Qwen3.5-27B full-teacher-prompt OOD benchmark

Date: 2026-08-31. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This benchmark measures the unadapted
`Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654` model on the
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
`79e05dcdf5ef35a0ecf6c68d9125d61128d99370cea81ba6a30f8a79f1892e3b`.
The 27B tokenizer vocabulary, merges, configuration, and chat template are
byte-identical to the pinned 4B and 9B benchmarks. The OOD suite was never used
to tune the prompt, score, request settings, or threshold.

## Paper-comparable result

The paper's primary aggregate is Mean-OOD pAUROC@20: the unweighted arithmetic
mean of the six independently computed per-dataset pAUROC@20 values.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.9371** | **0.8395** |
| Pooled examples | 6,395 | 0.9330 | 0.7969 |
| AgentDojo | 1,500 | 0.9829 | 0.9147 |
| CUA-SHADE-Arena | 955 | 0.9040 | 0.8150 |
| IaC | 1,443 | 0.9751 | 0.8859 |
| MLE-Sabotage | 148 | 0.9988 | 0.9939 |
| Rogue Eval | 1,499 | 0.9813 | 0.9338 |
| SHADE-Arena | 850 | 0.7803 | 0.4937 |

At the fixed 0.5 threshold, source-macro balanced accuracy is `0.8661`, macro
Brier score is `0.10691`, macro recall is `0.8382`, and macro FPR is `0.1061`.
The pooled balanced accuracy is `0.8671` and pooled Brier score is `0.10436`.
There are 4,035 unique continuous scores; 2,360 rows share a score with at
least one other row.

The matched dense-capacity curve is now `0.6175` at 4B, `0.6976` at 9B, and
`0.8395` at 27B. The 27B gain over 9B is `0.1418` pAUROC@20, larger than the
4B-to-9B gain of `0.0801`. Capacity improves all six sources, but SHADE-Arena
remains the largest low-FPR weakness. This fills an important part of the gap
between 9B and Kimi K3 (`0.9084`), while leaving a `0.0689` absolute gap and not
isolating parameter count from model-family or training differences.

At its canonical hosted-price proxy, this is a new observed frontier point. It
also point-estimate dominates the paper's Qwen3.5-27B SFT+RL row: pAUROC@20 is
`0.0085` higher and projected cost is `33.4%` lower. That is an interface-level
comparison, not a claim that base training supersedes the paper's trained
monitor: the paper generated rationales and ordinal scores under a short prompt,
whereas this run used an immediate binary logit under the full specification.

## Exact cost accounting

The exact rendered Qwen contexts contain 60,539,619 input tokens, or
9,466.7113 per evaluation. The maximum is 30,382 tokens, below the frozen
32,768-token engine limit. The selected-token interface produces exactly one
output token per row.

OpenRouter's public model catalog did not list Qwen3.5-27B in the dated price
snapshot, so there is no current OpenRouter route whose quote can be attached
to this result. For paper comparability, the canonical coordinate uses the
paper's Qwen3.5-27B hosted proxy of `$0.195/M` input tokens and `$1.56/M` output
tokens:

```text
input  = 9,466.7113 * $0.195 / 1000 = $1.846009 per 1K evaluations
output =       1.0 * $1.56  / 1000 = $0.001560 per 1K evaluations
total                                    $1.847569 per 1K evaluations
```

The frontier therefore records **`$1.8476/1K`**. This is the paper's dated
hosted proxy, not a current OpenRouter quote or the cost of the local Slurm run.
It assumes no prompt-cache or batch discount.

## Runtime and artifact audit

Slurm job `30977052` completed in 2 hours 18 minutes 15 seconds on one NVIDIA
RTX PRO 6000 Blackwell Server Edition using vLLM 0.24.0, PyTorch 2.11.0+cu130,
BF16 weights, one CPU, and 32 GB host RAM. The evaluator's measured interval
after engine construction was 8,052.02 seconds. The persistent text-only engine
used continuous batching, a one-token constrained response, and explicit
decision-token logprobs. The application submitted resume-safe 128-row windows;
vLLM dynamically scheduled at most 16 active sequences and 32,768 batched
prompt tokens inside each window.

All 6,395 rows are present exactly once, with the exact audited token total;
there were no missing logprobs, truncations, invalid decisions, duplicate IDs,
or backend failures. The balanced 12-row interface canary returned finite scores
for every source-label stratum.

The ignored result artifacts are:

- `results/tool_trajectory_monitoring/qwen35_27b_teacher_ood/predictions.jsonl`,
  SHA-256 `ddf302e31718a6850e914307b3047eb0417e0dc174787d7bfb9bf60e42b8f60a`;
- `results/tool_trajectory_monitoring/qwen35_27b_teacher_ood/canary_result.json`,
  SHA-256 `d7d6bef8ab7b3c50f209f5f3c0473d55e34fe763c0810961eabf04caf2a111ac`;
- `results/tool_trajectory_monitoring/qwen35_27b_teacher_ood/result.json`,
  SHA-256 `0f69cc3d8b839e6fe123d4a2e69d8209c3924560368fadb9705e9a0e06170778`.

The canonical cost--performance comparison is maintained in
[`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md).

- [Qwen3.5-27B model and revision history](https://huggingface.co/Qwen/Qwen3.5-27B)
- [OpenRouter model catalog search for Qwen3.5-27B](https://openrouter.ai/models?q=qwen3.5-27b)
- [Sinha et al., Appendix D.4 metric definition](https://arxiv.org/html/2605.29601v1#A4.SS4)
- [Sinha et al., Appendix E hosted proxy prices and cost accounting](https://arxiv.org/html/2605.29601v1#A5.SS4)
