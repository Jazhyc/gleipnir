# Price release models on the released stack, one request at a time

Status: current.
Date: 2026-11-18.

User decisions, 2026-11-16 and 2026-11-18, recorded while drafting the v1
technical report: price Gleipnir at self-hosted B200 break-even from the
**one-request-at-a-time throughput of the released serving image** on the
96-trajectory agent benchmark, and use the same basis for the OOD frontier
costs. This replaces the batched OOD wall time on the 4B v0.3 stack used by the
[2026-10-20 decision](price_gleipnir_release_models_at_self_hosted_b200_break_even.md).

**Rates.** One B200 at 100% utilization: `$7.99/h` on demand (what Runpod
charged) and `$3.44/h` committed use (CoreWeave's `$8.60` list price with its
largest published 60% discount). Cost = hourly rate / (prompt tokens/s ×
3,600). Report the on-demand cost until a provider serves the monitors; a
provider's price then replaces both figures.

**Stack.** The released image adds, on top of stock vLLM BF16 classification
mode: low-precision arithmetic (FP8 attention projections, FP4 MLP/GDN, MXFP8
attention), CUDA graphs (to 4,096 tokens for the 4B, 2,048 for the 27B),
last-token attention in the final layer, the shorter GDN host path, the GDN
kernel variant (4B only) and the Gigatoken tokenizer.

| | 4B | 27B |
| --- | ---: | ---: |
| Prompt tokens/s, one request at a time | 204,673 | 50,788 |
| USD per million prompt tokens, on demand / committed | 0.0108 / 0.0047 | 0.0437 / 0.0188 |
| USD per 1,000 OOD evaluations, on demand / committed | 0.0939 / 0.0404 | 0.4137 / 0.1781 |

Concurrency is not priced: 8 concurrent requests raise throughput by only 13%
(4B) and 5% (27B) while each request waits about nine times as long. Output
tokens are not priced.

Sources: report Table 1 and the serving-steps appendix table
([report sources](../../../outputs/reports/gleipnir_v1_technical_report/sources.md)),
the [frontier registry](../../findings/evaluation/tool_trajectory_ood_frontier.md#reference-table)
and the [rate sources](../../findings/evaluation/tool_trajectory_inference_economics.md).
