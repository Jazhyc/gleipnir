# Adopt last-query final-layer attention in the FP8 default

Date: 2026-10-15. Status: recorded in the decision log.

- The optimized serving default runs layer-31 attention only from each prefill
  request's last token. Scores are bitwise unchanged at C1. C1 latency drops
  6.4–10.3% from 256K to 1M, and same-node full320 c128 throughput rises 5.7%.
  See the [serving decision](b200_monitor_score_reference.md) and
  [finding](../../findings/serving/b200_last_layer_trim.md).
