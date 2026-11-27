# Gate the GDN context-parallel prefill route by length

Date: 2026-10-15.
Status: current.

- The FP8 default takes FlashInfer's non-CP GDN prefill below 8,192 tokens per
  call. In a same-server A/B through normal startup, quick64 c1 p50 is 2.3 ms
  lower and full320 c128 throughput is unchanged. Canary MAE is 0.0005. See the
  [finding](../../findings/serving/b200_gdn_route.md).
