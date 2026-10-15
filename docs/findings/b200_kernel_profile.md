# Kernel profile of the adopted last-query default

2026-10-15. These are torch-profiler traces of the adopted FP8 default with
last-query final-layer attention, on the same EU-RO-1 B200 as the
[1M baseline](b200_context_million.md). They are diagnostic only: profiler
overhead perturbs latency, so no timing claim or recipe change follows. See
[the experiment](../../experiments/b200_kernel_profile/README.md).

## Runs

- `profile01` failed before server readiness. The selected recipe's archived
  diagnostic sources had not been restored, and the guard rejected the
  synchronized copy. The profiler now restores them, as the regular runners do.
- `profile02` traced the native-RoPE cases (C1 1K/8K/32K/256K and full320 c128).
  A wrong `retire()` call then left its server running. That server was retired
  by identity check before `profile03` traced the 1M YaRN case.

Native canaries reproduced the default exactly. Traces and SHA-256 hashes are
under `results/b200_kernel_profile/`.

## Where GPU time goes

Shares are of summed kernel time. "Idle" is the share of the GPU window with
no kernel running.

| Case | GPU idle | MXFP8 attention | FP4 GEMM | GDN (conv, recurrence, norm) | Memory copies | Norm/RoPE | FP8 GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 1K | 81.3% | 1% | 26% | 28% | 15% | 4% | 4% |
| C1 8K | 13.2% | 6% | 37% | 25% | 10% | 6% | 5% |
| C1 32K | 3.9% | 17% | 34% | 21% | 8% | 6% | 4% |
| C1 256K | 2.0% | 60% | 17% | 10% | 4% | 3% | 2% |
| C1 1M (YaRN) | 0.6% | 84% | 6% | 4% | 1% | 1% | 1% |
| full320 c128 | 9.5% | 9% | 37% | 23% | 9% | 6% | 5% |

**Long contexts are attention-bound.** The cuDNN DSL MXFP8 SDPA kernel
(D=256) delivers about 2.2 PFLOP/s at 256K and 2.05 PFLOP/s at 1M, counting
seven full causal layers. That is roughly half of nominal dense FP8 tensor
throughput, measured while the power cap held the SM clock near 1.45–1.6 GHz.
MXFP8 history packing (`_produce`) adds 1.7% at 1M.

**Mid-length and batched work is GEMM- and GDN-bound.**
- The FP4 SwiGLU gate/up GEMM alone is 22% at 32K and 24% at c128. The FROST
  block-scaled FP4 GEMMs (MLP down and GDN projections) add 12–13%.
  Approximate gate/up throughput at 32K is about 3 PFLOP/s, which is likely
  well below FP4 peak.
- Within GDN, the width-4 `_causal_conv1d_fwd_kernel` costs more than the
  FlashInfer recurrence: 9.2 vs 6.6 ms at 32K, 370 vs 652 ms at c128, and
  387 vs 288 ms at 1M. With `_fused_post_conv_kernel` and the gated
  `layer_norm_fwd`, the non-recurrent GDN kernels are about 11% at 32K.
- Three ATen `direct_copy` launches per GDN layer per step (72 per 32K step)
  are about 5% at 32K and c128.

**Short requests are host-bound.** At 1K, one 1,024-token step takes about
46 ms of host time under the profiler, against 8.3 ms of GPU kernels. Eager
host work inside splitting/custom ops dominates: about 14.4 ms in 24 GDN core
calls, 14.6 ms in 112 FROST FP4 linear calls, and 7.7 ms in FP4 SwiGLU and
packing ops. There are no CUDA-graph launches at this size (1,104 individual
launches). A 0.24-era large-prefill graph capture
([finding](b200_prefill_graph_capture.md)) cut launches 55% but regressed c1
and c128, so graphs alone are not the indicated fix.

## Optimization candidates

These are candidates only; none was tested.

1. **MXFP8 attention kernel configuration**, for long contexts. The serving plan
   fixes `cga=1` and `split_kv=1`. Screening 2-CTA/cluster and tile variants of
   the same D=256 kernel at 32K query chunks over 64K–1M histories is the
   largest lever at 256K and above.
2. **FP4 SwiGLU GEMM efficiency**, for batched and mid-length work. Check its
   per-M tile choice and achieved throughput against an isolated benchmark.
3. **GDN pre/post fusion.** Fuse the causal conv1d, post-conv and gated norm,
   and remove the per-layer contiguity copies. This plausibly gives 5–10% at
   mid lengths and c128, and should keep arithmetic close to exact.
4. **Host dispatch** for short requests: lower-overhead GDN/FROST wrappers
   rather than broader graph capture.

Category assignment uses ordered kernel-name patterns in
`experiments/b200_kernel_profile/analyze.py`. FLOP-rate figures are
approximate and are not profiler counters.
