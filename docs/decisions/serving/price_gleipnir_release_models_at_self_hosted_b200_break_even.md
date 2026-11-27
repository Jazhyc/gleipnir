# Price Gleipnir release models at self-hosted B200 break-even

Date: 2026-10-20.
Status: superseded by [Price release models on the released stack, one request at a time](price_release_models_on_the_released_stack.md).

- User decision: Gleipnir 4B v0.3 and Gleipnir 27B are priced at self-hosting
  break-even, one B200 at `$7.99/h` at 100% utilization. The reference is
  this benchmark's workload: measured optimized-serving wall time for all
  6,395 OOD rows × hourly rate, per 1,000 rows.
- Gleipnir 4B v0.3 costs `$0.0850/1K` (226K prompt tokens/s, `$0.0098/M`).
  Gleipnir 27B costs `$0.3896/1K` (53.9K tokens/s under its teacher prompt,
  `$0.0412/M`). The frontier becomes Gleipnir 4B, pplx-decider v1.2 and
  Gleipnir 27B.
- Other rows keep list prices or hosted proxies, which include provider
  utilization and margin.
- User decision (same day): each release model also gets a committed-GPU
  row at `$3.44/h`, CoreWeave's published committed-use ceiling (60% off
  $8.60), as an inference-provider estimate. The plot shows on-demand and
  committed-GPU frontiers separately. Sources are under
  [B200 hourly rate sources](../../findings/evaluation/tool_trajectory_inference_economics.md#b200-hourly-rate-sources).
- For a per-token endpoint, throughput is flat to about 32K tokens and falls
  beyond it (27B optimized: about 1.4× the per-token cost at 128K, 2.2× at
  256K and 6× at 1M), so long contexts would need tiered pricing.
