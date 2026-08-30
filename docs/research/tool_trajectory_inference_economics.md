# Tool-trajectory monitor inference economics

Status: exact Qwen and Kimi evaluation accounting, updated 2026-08-31. This note prices
monitor inference only. It does not price Kimi annotation, Qwen training, or
self-hosted total cost of ownership.

## Paper accounting convention

Sinha et al. report uncached marginal USD per 1,000 monitor evaluations over
their six-source, 6,395-row OOD suite. For model `m`, their calculation is:

```text
cost_per_1k = (mean_input_tokens * input_USD_per_Mtok
               + mean_output_tokens * output_USD_per_Mtok) / 1000
```

Prompted API monitors use saved provider usage. Trained open-weight monitors use
the base tokenizer to count the rendered short prompt and saved completion.
Provider-billed reasoning tokens count as output. The main estimate deliberately
excludes prompt caching, batch discounts, SFT/RL, queueing, throughput, and
self-hosted infrastructure. Preserve those exclusions in a paper-comparable
cost-performance plot and show amortized or self-hosted costs separately.

Selected paper reference rows are below. The complete 17-row paper table plus
all completed Gleipnir evaluations is maintained in
[`tool_trajectory_ood_frontier.md`](tool_trajectory_ood_frontier.md).

| Monitor | Mean input | Mean output | Input $/M | Output $/M | $/1K evaluations | Mean-OOD pAUROC@20 | Mean-OOD AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B SFT+RL | 8,411.0 | 705.6 | 0.03 | 0.15 | 0.358 | 0.684 | 0.884 |
| Qwen3.5-27B SFT+RL | 8,411.0 | 727.5 | 0.195 | 1.56 | 2.775 | 0.831 | 0.948 |
| Gemini 3.1 Pro prompted | 13,995.1 | 3,769.0 | 2.00 | 12.00 | 73.218 | 0.940 | 0.973 |

The paper's Qwen monitors generate rationales before an ordinal answer.
Gleipnir's main base-model baselines instead normalize the literal `0` and `1`
logits at an immediate decision boundary. A matched Qwen3.5-9B ablation also
generates an immediate 0--10 integer without a rationale. Both interfaces have
negligible output-token charges, but every monitor must still prefill the long
trajectory.

## Hosted price snapshots

The 2026-08-30 hosted-price snapshot used here is:

| Model | Pricing basis | Input $/M | Output $/M | Cache-read $/M |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-4B | Paper's Artificial Analysis hosted proxy; no current OpenRouter route | 0.03 | 0.15 | not assumed |
| Qwen3.5-9B | OpenRouter `qwen/qwen3.5-9b` | 0.10 | 0.15 | not assumed |
| Qwen3.5-27B | Paper's hosted proxy; no current OpenRouter route | 0.195 | 1.56 | not assumed |
| Kimi K3 on Makora | OpenRouter `moonshotai/kimi-k3` | 2.55 | 12.75 | 0.256 |

These are mutable hosted prices, not claims about underlying GPU cost. Record a
new dated row rather than silently replacing this snapshot when prices change.
Use uncached input in the main comparison. The current Kimi policy prefix did
not produce cross-row cache reuse in the paid canary, and the paper also omits
cache discounts.

## Gleipnir Qwen3.5 exact accounting

The completed binary 4B, 9B, and 27B runs share a byte-identical tokenizer and
chat template and therefore use the same 60,539,619 rendered input tokens, or
9,466.7113 per evaluation. All generate exactly one constrained output token.
The 9B ordinal ablation uses a longer instruction, averaging 9,624.7113 input
tokens, and records two output tokens per row because vLLM appends EOS after a
one-digit score and `10` itself takes two tokens. Their exact paper-comparable
coordinates are:

| Model | Input price $/M | Output price $/M | Input USD / 1K evaluations | Output USD / 1K evaluations | Total USD / 1K evaluations | Mean-OOD pAUROC@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B base | 0.03 | 0.15 | 0.284001 | 0.000150 | **0.284151** | 0.6175 |
| Qwen3.5-9B base | 0.10 | 0.15 | 0.946671 | 0.000150 | **0.946821** | 0.6976 |
| Qwen3.5-9B base, immediate ordinal | 0.10 | 0.15 | 0.962471 | 0.000300 | **0.962771** | 0.4858 |
| Qwen3.5-27B base | 0.195 | 1.56 | 1.846009 | 0.001560 | **1.847569** | 0.8395 |

The 4B price is the paper's Artificial Analysis hosted proxy. OpenRouter did
not list Qwen3.5-4B on 2026-08-30, so it must not be described as an OpenRouter
quote. The 9B row uses OpenRouter's dated route. The 27B row likewise uses the
paper's dated hosted proxy because OpenRouter did not list a current 27B route.
The matched dense-capacity curve rises by `0.0801` pAUROC@20 from 4B to 9B and
by `0.1418` from 9B to 27B. The ordinal ablation costs 1.7% more than the binary
9B run while losing `0.2118` pAUROC@20, so it is strictly dominated.

## Direct Kimi K3 monitor exact tokenization

Treat Kimi K3 with the frozen full binary teacher prompt as a prompted-monitor
baseline, not as part of Qwen's deployment cost. The exact paper-author OOD
input artifacts are tokenized below; this supersedes the earlier ratio-based
planning estimate.

The earlier planning calculation used the paired 3,012-row held-out ID set:

- the full Kimi request averages 11,075.58 predicted Makora prompt tokens;
- the implemented compact Qwen prompt averages 11,495.87 Qwen tokens;
- their paired aggregate ratio is 0.96344.

Applying that ratio to the paper's 8,411-token Qwen OOD mean suggested an
8,100--8,400-token planning interval. Exact materialization now uses the six
`filtered_eval/*.parquet` inputs from
`adityaasinha28/deliberative-monitor-pipeline` revision `2f10f4e...`, renders
the frozen full binary teacher prompt, and tokenizes it with
`moonshotai/Kimi-K3` revision `a590ce...`. The provider projection adds the
21-token chat wrapper observed on every one of 3,312 prior Fireworks requests.
Ten completion tokens per row match those requests exactly; a `max_tokens` cap
is not itself billed.

The exact provider projection is **53,222,720 input tokens**, or 8,322.55 per
row. At Fireworks' 2026-08-30 OpenRouter rates of `$3.00/M` input and `$15.00/M`
output, with the observed ten completion tokens per row:

| Source | Rows | Provider input tokens | Mean input | Input USD | Output USD | Total USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUA-SHADE-Arena | 955 | 9,770,065 | 10,230.43 | `$29.3102` | `$0.1433` | **`$29.4534`** |
| MLE-Sabotage | 148 | 1,295,072 | 8,750.49 | `$3.8852` | `$0.0222` | **`$3.9074`** |
| AgentDojo | 1,500 | 2,268,781 | 1,512.52 | `$6.8063` | `$0.2250` | **`$7.0313`** |
| IaC | 1,443 | 19,495,920 | 13,510.69 | `$58.4878` | `$0.2165` | **`$58.7042`** |
| Rogue Eval | 1,499 | 5,133,854 | 3,424.85 | `$15.4016` | `$0.2249` | **`$15.6264`** |
| SHADE-Arena | 850 | 15,259,028 | 17,951.80 | `$45.7771` | `$0.1275` | **`$45.9046`** |
| **Total** | **6,395** | **53,222,720** | **8,322.55** | **`$159.6682`** | **`$0.9593`** | **`$160.6274`** |

This is **`$25.1177 per 1,000 evaluations`** at uncached list price. The exact
ignored prompt manifest is reconstructed with:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_teacher_ood_cache
```

The completed Kimi K3 evaluation reports 53,222,737 prompt tokens and 63,950
completion tokens across all 6,395 successful rows. The 17-token difference
from the local prompt projection is negligible. Fireworks reported 482,909
cache-read tokens (`0.91%`) and `$157.73037` in the retained successful-row
usage records. This is not the paper-comparable price coordinate: it includes a
small cache discount and excludes the additional charge from five paid HTTP-200
null responses whose failure usage was not retained. Preserve the uncached
`$160.6274` suite projection above as the primary cost, and treat actual account
spend as greater than `$157.73037` but not exactly reconstructable from the
generic cache.

The prepared author inputs contain 1,499 Rogue Eval and 850 SHADE-Arena rows;
the separate released prediction artifact has 1,500 and 849. Both have 6,395
total rows and identical aggregate class counts. The input artifact is the
correct source for prompt materialization. The repository declares no license,
so keep trajectories and rendered prompts internal and do not redistribute
them.

Do not substitute the separate `$214.39` training-cache annotation estimate:
annotation is a one-time data-construction cost, whereas this section prices
Kimi evaluating every deployment trajectory.

## Exact accounting gate

Before publishing the cost-performance plot:

1. Reuse the now-materialized frozen 6,395 OOD rows for every monitor.
2. Render each monitor's exact prompt and chat template.
3. Count Qwen prompts with the pinned Qwen3.5-9B tokenizer and retain the actual
   selected-token evaluation output count.
4. The Kimi prompts are exactly tokenized locally; prefer saved OpenRouter usage
   after the paid evaluation and reconcile it with the 21-token wrapper model.
5. Record actual model/provider, ordinary and cached input tokens, output and
   reasoning tokens, failures, retries, and price-snapshot time.
6. Report uncached list-price cost as the paper-comparable primary result, with
   cache/batch and self-hosted sensitivities separately.

The frozen Kimi evaluation is complete; its paper-comparable Mean-OOD
AUROC/pAUROC@20 is `0.9616/0.9084`. See
[`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md)
for the per-source result and serving-failure audit. Do not pair the provisional
Qwen cost coordinate with an OOD performance claim until that evaluation
completes.

## Specialized prefill-only serving backlog

This is future systems work, not part of the first monitor campaign. Gleipnir is
closer to long-document classification than chat generation: it needs one
forward pass and two final-position logits, with no autoregressive loop. A
provider optimizing specifically for this contract could:

1. expose `trajectory -> P(problematic)` rather than a chat-completions API;
2. take the final hidden state and project only onto the two literal LM-head
   rows, avoiding a 248,320-way vocabulary projection and softmax;
3. omit sampling, detokenization, streaming, stop checks, and decode scheduling;
4. for independent trajectories, avoid retaining a persistent generation KV
   cache and use temporary layer-local prefill state;
5. pack variable-length prompts without padding, bucket by length, and batch by
   total prompt tokens rather than request count;
6. load Qwen3.5 text-only, use the pinned FLA gated-delta kernels and an exact
   variable-length attention kernel, and test FP8 against the BF16 reference;
7. keep 9B on one GPU to avoid tensor-parallel communication and scale replicas
   with data parallelism;
8. for repeated scoring of a growing live trajectory, do the opposite of item
   4: retain session-bound attention and gated-delta state so each appended
   segment is processed once rather than replaying the entire prefix.

The relevant self-hosted cost object is evaluations per second, not interactive
decode tokens per second:

```text
self_hosted_USD_per_1k = 1000 * accelerator_USD_per_hour
                         / (3600 * evaluations_per_second)
```

Benchmark this under the real 8--9K-token length distribution and a declared
latency/concurrency target. API list price is a market price influenced by
provider utilization, quantization, batching, hardware, routing, margin, and
subsidy; it is not directly proportional to parameter count or single-request
speed.

## Sources

- [Sinha et al., Appendix E cost accounting](https://arxiv.org/html/2605.29601v1#A5)
- [Sinha et al., Qwen3.5-4B hosted proxy price](https://arxiv.org/html/2605.29601v1#A5.SS4)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b/pricing)
- [OpenRouter model catalog search for Qwen3.5-27B](https://openrouter.ai/models?q=qwen3.5-27b)
- [OpenRouter Kimi K3 pricing](https://openrouter.ai/moonshotai/kimi-k3-20260715)
- [Qwen3.5-27B model and text-only serving notes](https://huggingface.co/Qwen/Qwen3.5-27B)
- [vLLM classification usage](https://docs.vllm.ai/en/latest/models/pooling_models/classify/)
- [FlashAttention reference implementation](https://github.com/Dao-AILab/flash-attention)
- [`tool_trajectory_teacher_cost_estimate.md`](../findings/tool_trajectory_teacher_cost_estimate.md)
