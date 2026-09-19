# Qwen3.5-35B-A3B mixed-provider full-teacher-prompt OOD benchmark

Date: 2026-09-19. Status: completed on all 6,395 strict OOD rows.

## Frozen design and provider handoff

The unadapted `Qwen/Qwen3.5-35B-A3B` tokenizer/model route uses tokenizer
revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, OpenRouter model ID
`qwen/qwen3.5-35b-a3b`, the unchanged six-source OOD artifact, and the full
binary teacher specification. The template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`;
the input SHA-256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`.
Thinking was disabled. The provider generated exactly `Prediction:0` or
`Prediction:1` in four tokens per row; we normalized the terminal literal
`1` and `0` token logprobs. This differs from the dense Qwen3.5-27B's
one-token prefilled decision boundary.

Darkbloom FP4 completed 4,545 OOD rows, then its serving throughput slowed.
The provider handoff was triggered by speed, without inspecting OOD scores,
labels, or metrics. The exact 4,545-row checkpoint was pinned by SHA-256 and
only 1,850 missing IDs were sent to Parasail FP8. Both routes were pinned with
fallbacks disabled. Parasail's separate 100-row non-OOD canary passed before
launch, and a label-blind request for the longest 30,382-token OOD prompt
passed its 262,144-token context gate. The mixed contract's SHA-256 is
`ac8ac3e74445777a37424e20e501343b8e58fe77dd19c8068fbbbeb6e7951f91`.

## Result

The paper's primary metric averages the six independently calculated raw
normalized pAUROC@20 values, without source-size weighting.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.9050** | **0.7580** |
| Pooled examples | 6,395 | 0.8421 | 0.6032 |
| AgentDojo | 1,500 | 0.9654 | 0.8499 |
| CUA-SHADE-Arena | 955 | 0.8934 | 0.7690 |
| IaC | 1,443 | 0.8991 | 0.6688 |
| MLE-Sabotage | 148 | 0.9955 | 0.9776 |
| Rogue Eval | 1,499 | 0.9103 | 0.7497 |
| SHADE-Arena | 850 | 0.7662 | 0.5328 |

At threshold 0.5, source-macro balanced accuracy is `0.7845`, recall
`0.7584`, false-positive rate `0.1894`, and Brier score `0.15741`. The pooled
values are `0.7617`, `0.7439`, `0.2204`, and `0.17174`, respectively.
There are 908 unique scores and 5,487 rows tied with at least one other row.

The sparse hosted baseline is `0.0815` pAUROC@20 below the matched dense 27B
baseline (`0.8395`) and `0.0243` below the trained mixed 4B student (`0.7823`).
Its source distribution is entangled with the provider handoff: Darkbloom
served all AgentDojo, CUA-SHADE-Arena, IaC, and MLE-Sabotage rows plus 499
Rogue Eval rows; Parasail served the other 1,000 Rogue Eval rows and all 850
SHADE-Arena rows. This result cannot isolate sparse architecture, model
training, endpoint quantization, or provider implementation as the cause of
the observed gap. It is not a clean single-provider model-capacity curve.

On 100 paired non-OOD canary prompts, Parasail's pooled AUROC was `0.9036`
versus Darkbloom's `0.8790`, with 94 versus 67 unique scores. Pearson score
correlation was `0.9467`, Spearman correlation `0.9647`, mean absolute score
difference `0.0695`, and fixed-0.5 decision agreement `90%`. This supports
the shared interface while confirming meaningful route-level score shifts.

## Cost and audit

The exact local prompt audit totals 60,539,619 tokens; OpenRouter reported
60,526,829 prompt tokens and 25,580 completion tokens over the 6,395 OOD
responses. Frozen uncached route prices on 2026-09-19 were `$0.08/M` input and
`$0.75/M` output for Darkbloom, and `$0.15/M` input and `$1.00/M` output for
Parasail. Applied to the actual per-provider usage, the cost coordinate is:

| Provider | OOD rows | Prompt tokens | Output tokens | List-price USD |
| --- | ---: | ---: | ---: | ---: |
| Darkbloom FP4 | 4,545 | 39,591,858 | 18,180 | 3.180984 |
| Parasail FP8 | 1,850 | 20,934,971 | 7,400 | 3.147646 |
| **Combined OOD** | **6,395** | **60,526,829** | **25,580** | **6.328629** |

The canonical frontier coordinate is **`$0.9896/1K`** evaluations, excluding
canaries, failed interface attempts, and retries under the registry contract.
The retained OOD rows' provider-reported cost sums to `$6.265343`; the
Parasail canary adds `$0.173930` reported or `$0.175687` at frozen list price.
The mixed OOD plus Parasail canary ceiling was `$6.504316`, under the frozen
`$6.60` stop condition. The earlier Darkbloom canary and its failed partial
prefill experiment are separate operational costs, not part of that ceiling.

All 6,395 IDs occur once, all scores are finite and in `[0, 1]`, all responses
use the expected model and provider, and all terminal decisions include both
literal logprobs. Saved successful rows report 6,129 first-attempt successes
and 266 successes after retries (216 second attempts, 22 third, 20 fourth,
8 fifth). These are response-attempt counts; charged failed attempts are not
recoverable from retained successful-row usage. The Parasail continuation's
invocation took `517.39` seconds including local validation; it completed the
1,850 missing rows in 29 checkpoint batches at concurrency 16.

The ignored result artifacts are:

- Darkbloom partial `results/tool_trajectory_monitoring/qwen35_35b_a3b_darkbloom_teacher_ood/predictions.jsonl`:
  SHA-256 `d726fbbcdb8b5a58532722f65dad929dad488a39e531987296a485db9fa515d8`;
- mixed `results/tool_trajectory_monitoring/qwen35_35b_a3b_parasail_teacher_ood/predictions.jsonl`:
  SHA-256 `58a80e79ad20322347d7d351d863810336468beff16308d710e8f1ba485b4859`;
- mixed `results/tool_trajectory_monitoring/qwen35_35b_a3b_parasail_teacher_ood/result.json`:
  SHA-256 `e0bf7f1dfe6cbc0df8df91631c5fd23f13e483bc814407ce16ae1549c92a44f7`.

The canonical comparison is in
[`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md).
