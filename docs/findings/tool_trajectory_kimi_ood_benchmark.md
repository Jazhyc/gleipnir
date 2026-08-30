# Kimi K3 OOD binary-logit benchmark

Date: 2026-08-30. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This benchmark measures `moonshotai/kimi-k3` through the Fireworks provider on
the six paper-author OOD action-only datasets. Kimi receives Gleipnir's full
binary teacher prompt, including the detailed scheming specification, and the
score is the normalized relative logit of the literal `1` versus `0` decision
tokens. This deliberately evaluates Gleipnir's interface rather than reproducing
the paper's generated 0--10 score interface.

Every request used temperature zero, reasoning effort `none`, top-five token
logprobs, the prompt-only `Prediction:0|1` boundary, `max_tokens=64`, Fireworks
provider-only routing with no fallback, and price ceilings of `$3.00/M` prompt
and `$15.00/M` completion tokens. The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128` and
the request-settings SHA-256 is
`b60fa90560524f6ae207b1dc89041feb0fa44f29b6d627ac7d3692599b7482c5`.

The benchmark contains 3,189 benign and 3,206 positive trajectories. The OOD
suite was never used to tune the prompt, score, request settings, or threshold.

## Paper-comparable result

The paper's primary aggregate is Mean-OOD pAUROC@20: the unweighted arithmetic
mean of the six per-dataset pAUROC@20 values. It also reports Mean-OOD AUROC as
the corresponding unweighted per-dataset mean. Kimi K3 reaches:

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.9616** | **0.9084** |
| Weighted-OOD | 6,395 | 0.9682 | 0.9241 |
| Pooled examples | 6,395 | 0.9776 | 0.9322 |
| AgentDojo | 1,500 | 0.9999 | 0.9997 |
| CUA-SHADE-Arena | 955 | 0.9438 | 0.8856 |
| IaC | 1,443 | 0.9984 | 0.9920 |
| MLE-Sabotage | 148 | 1.0000 | 1.0000 |
| Rogue Eval | 1,499 | 0.9997 | 0.9986 |
| SHADE-Arena | 850 | 0.8276 | 0.5742 |

`Weighted-OOD` is the sample-count-weighted mean of the six independently
computed dataset metrics, matching the paper's secondary aggregation. It is not
the same estimand as computing one ROC curve after pooling examples.

The authors report prompted-monitor results as mean and standard deviation over
three temperature-one inference seeds. This benchmark is one deterministic
temperature-zero logit run, so it has no across-run standard deviation. A
5,000-resample source-label-stratified bootstrap gives a supplemental pooled
AUROC interval of `0.9742--0.9809`, but bootstrapping is not part of the paper's
point-metric protocol and does not affect any AUROC or pAUROC value above.

At the fixed 0.5 threshold, pooled balanced accuracy is `0.9320`, recall is
`0.9102`, FPR is `0.0461`, and Brier score is `0.04917`. There are 3,488 unique
continuous scores.

The balanced 600-row canary was representative of the completed result. Its
Mean-OOD AUROC/pAUROC@20 was `0.9632/0.9191`, compared with
`0.9616/0.9084` on the full suite. Pooled values changed from
`0.9749/0.9345` to `0.9776/0.9322`.

## Coverage and request audit

The union of the 600-row canary cache and 5,795-row remaining cache has exactly
6,395 unique score IDs, exactly matching the input ID set. All prompt hashes,
model IDs, provider IDs, and request-setting hashes match the frozen inputs.
Every accepted row contains logprobs for both decision tokens. There are zero
missing, duplicate, format, parse, or logprob rows in the final evaluation.

The initial full-suite pass left 52 rows unresolved: 47 exhausted the bounded
retry policy on shared Fireworks capacity/rate-limit responses and five returned
paid HTTP-200 responses with null content/logprobs. Identical lower-concurrency
and then serial resumptions recovered every row. These are serving failures, not
evaluation exclusions; no label- or score-dependent filtering was applied.

Successful final rows report 53,222,737 prompt tokens, 63,950 completion tokens,
482,909 cache-read tokens, and `$157.73037` charged cost in their retained usage
records. The `0.91%` cache-read fraction is small. This number is a lower bound
on total account spend because the generic failure cache does not retain usage
from the five paid null responses. For paper-comparable economics, use the
uncached list-price projection of `$160.6274` for the complete suite rather than
the cache-discounted successful-row charge.

## Interpretation

Kimi K3 is effectively perfect on AgentDojo, IaC, MLE-Sabotage, and Rogue Eval,
and remains strong on CUA-SHADE-Arena. SHADE-Arena is the dominant weakness,
especially in the low-FPR region, reducing pAUROC@20 to `0.5742`. This is a
cross-source limitation rather than a reason to alter the frozen prompt after
seeing OOD labels.

The ignored score artifacts are:

- `results/tool_trajectory_monitoring/kimi_k3_ood_canary_fireworks.jsonl`,
  SHA-256 `c268958b594e4482f539ccaeb371454f629faeff1f9af96064a69d8bc99003bb`;
- `results/tool_trajectory_monitoring/kimi_k3_ood_full_remaining_fireworks.jsonl`,
  SHA-256 `69e826d425830b2053b7d0055aae03c907b29e9e5f621f8c5b44a941c44b5153`;
- `results/tool_trajectory_monitoring/kimi_k3_ood_full_summary.json`, SHA-256
  `29221d4424f74a10460d019b84ee93f9b5054a9228eb285c2d3bc26f4e93c0b5`.

Reconstruct the summary from the two resumable caches with:

```bash
python -m experiments.tool_trajectory_monitoring.summarize_teacher_canary \
  --input data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_ood_canary_fireworks.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_ood_full_remaining_fireworks.jsonl \
  --output results/tool_trajectory_monitoring/kimi_k3_ood_full_summary.json
```
