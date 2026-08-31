# Kimi K2.6 Inceptron OOD binary-logit benchmark

Date: 2026-09-01. Status: completed on all 6,395 strict OOD rows.

## Result

Unadapted `moonshotai/kimi-k2.6`, served exclusively by Inceptron's dated
`moonshotai/kimi-k2.6-20260420` endpoint, reaches **0.9194 Mean-OOD
pAUROC@20** and **0.9751 Mean-OOD AUROC** under Gleipnir's frozen full binary
teacher prompt. It uses no visible reasoning and scores the normalized
top-logprobs of the terminal literal `0` and `1` tokens.

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.9751** | **0.9194** |
| Weighted-OOD | 6,395 | 0.9772 | 0.9257 |
| Pooled examples | 6,395 | 0.9675 | 0.8931 |
| AgentDojo | 1,500 | 0.9995 | 0.9975 |
| CUA-SHADE-Arena | 955 | 0.9419 | 0.8731 |
| IaC | 1,443 | 0.9932 | 0.9685 |
| MLE-Sabotage | 148 | 1.0000 | 1.0000 |
| Rogue Eval | 1,499 | 0.9874 | 0.9530 |
| SHADE-Arena | 850 | 0.9286 | 0.7244 |

At the fixed 0.5 threshold, pooled balanced accuracy is `0.8828`, recall is
`0.9554`, FPR is `0.1897`, and Brier score is `0.08520`. There are 3,818 unique
scores; 3,501 rows belong to a repeated-score group and the largest tie has 66
rows. The ties do not prevent strong ranking performance but are materially
more common than a nominally continuous API margin might suggest.

## Frozen design and canary

The benchmark reuses the complete Kimi K3 input artifact, including all 6,395
paper-author strict-OOD row IDs and the byte-identical full monitoring
specification. The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`.
The model, provider, prompt, score boundary, request settings, full-suite
membership, and interpretation thresholds were frozen before the first paid
request in
[`kimi_k26_ood_benchmark.json`](../../experiments/tool_trajectory_monitoring/kimi_k26_ood_benchmark.json).

The first 12 interleaved source-label rows passed provider identity, terminal
digit, two-literal-logprob, price, and zero-reasoning gates. Inceptron
consistently prepended one ASCII space to its otherwise exact prediction line;
the pre-existing parser already accepts surrounding whitespace. This
label-blind provider-format amendment was recorded before scaling and changed
neither the prompt nor score extraction.

The complete frozen 600-row canary reached `0.936750` Mean-OOD pAUROC@20 and
`0.978933` Mean-OOD AUROC. The full result is lower by `0.017346` and
`0.003826`, respectively, but preserves the canary's high-quality conclusion.
The canary was descriptive and did not select whether or how to run the full
suite.

## Frontier comparison

The result dominates every recorded open-weight point above Qwen3.5-27B and
becomes the highest-quality open-weight point on the observed
cost--performance frontier.

| Comparison | pAUROC@20 | Paper-priced uncached USD / 1K | K2.6 change |
| --- | ---: | ---: | ---: |
| Kimi K2.6, full prompt and binary logits | **0.9194** | **7.9110** | — |
| Paper Kimi-K2.6 SFT | 0.8410 | 10.1750 | `+0.0784`, 22.3% cheaper |
| Kimi K3, full prompt and binary logits | 0.9084 | 25.1177 | `+0.0110`, 68.5% cheaper |
| Paper Claude Sonnet 4.6 prompted | 0.9290 | 58.8500 | `-0.0096`, 86.6% cheaper |

The K2.6-over-K3 macro gain is not uniform. K2.6 loses pAUROC@20 on AgentDojo
(`-0.0022`), CUA-SHADE (`-0.0125`), IaC (`-0.0236`), and Rogue Eval
(`-0.0456`), ties on MLE-Sabotage, and gains `+0.1502` on SHADE-Arena. The
source-balanced improvement is therefore driven entirely by correcting K3's
large SHADE weakness. K3 remains stronger on pooled pAUROC@20
(`0.9322` versus `0.8931`) and at the fixed threshold. The defensible claim is
macro-OOD frontier dominance under the registry's frozen estimand, not uniform
per-source superiority.

The comparison with the paper's K2.6 SFT point changes both prompt and output
interface. It shows that the full-specification binary-logit deployment point
dominates the published trained rationale-and-ordinal point; it is not a
matched causal test of whether SFT helps under a fixed interface.

## Cost and serving audit

All accepted rows report 53,145,980 prompt tokens and 25,580 completion tokens,
exactly four completion tokens per row, with zero reasoning tokens. The paper's
Table 3 fixes Kimi-K2.6 at `$0.95/M` input and `$4.00/M` output. Applying those
rates to our saved token usage gives the primary paper-comparable cost:

```text
input  = 53,145,980 * $0.95/M = $50.488681
output =     25,580 * $4.00/M = $ 0.102320
total                              $50.591001
per 1,000 evaluations              $ 7.911024
```

At Inceptron's frozen route prices of `$0.53/M` input and `$3.39/M` output, the
same usage is `$28.254086`, or `$4.418153/1K`, before caching. Inceptron
reported 14,143,360 cached input tokens, or 26.61% of prompt usage, and
OpenRouter reported `$22.930851` in successful-row charges. These are useful
operational costs, but the canonical frontier uses the paper-priced uncached
`$7.9110/1K` coordinate.

The 600-row canary took 333.2 seconds. The remaining 5,795-row shard took
2,645.8 seconds at concurrency eight. A concurrency-16 canary pass temporarily
exhausted Inceptron's shared pool: three unresolved 429 rows completed unchanged
at concurrency two. The full shard used concurrency eight and completed without
an unresolved row. Accepted records required one through seven HTTP attempts;
the scorer, prompt, and inclusion rule never changed.

## Coverage and artifacts

The union has exactly 6,395 unique IDs matching the frozen input, one provider,
one requested model, one selected dated endpoint, one prompt hash, and one
request-settings hash. Every response finished normally as one prediction
line, included both literal-token logprobs, and reported zero reasoning tokens.

- 600-row score cache:
  `2fab4999c50dc4978102d35b950e370c753cdd600256911f185a29dd0df2810d`
- 5,795-row remaining cache:
  `034449dc33c2f2896b12e6ba35a9a9f6d3967c40b4a2b8d0868da6d5382241c5`
- 600-row summary:
  `30557c0c8193ad0ce346240615126ae7548b146d0f9d1486b7b946ad99e20588`
- Full summary:
  `56a32b0e3a7292641f3393dfd0b818aa67b77775b79be47c5905f2d8ebc35905`

Reconstruct the ignored full summary with:

```bash
python -m experiments.tool_trajectory_monitoring.summarize_teacher_canary \
  --input data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k26_inceptron_ood_canary.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k26_inceptron_ood_full_remaining.jsonl \
  --output results/tool_trajectory_monitoring/kimi_k26_inceptron_ood_full_summary.json
```
