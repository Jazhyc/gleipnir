# Adopt the lean GDN host path with a strided gated norm

Date: 2026-10-15. Status: recorded in the decision log.

- The FP8 default runs the [lean GDN prefill host path](../../../experiments/b200_gdn_fast/README.md)
  with unchanged kernels. Its gated RMS norm reads the output gate in place
  instead of cloning it. Scores are bitwise unchanged at 1K–1M. In paired
  in-server A/Bs, short C1 is about 3 ms faster, 16K/32K C1 is 6.8%/4.2% faster,
  and full320 c128 throughput is about 4.5% higher. Startup verifies both paths
  ran.
