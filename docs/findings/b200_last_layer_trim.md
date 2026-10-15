# Last-query attention in the final layer

2026-10-15. Running layer-31 attention only for each request's final query,
while still writing every token's K/V, leaves every monitor score **bitwise
unchanged**. It reduces C1 latency by 6.4% at 256K, 8.7% at 512K and **10.3% at
1M** (43.08 s to 38.64 s) on the same B200 as the
[1M baseline](b200_context_million.md). Below 32K the change is within noise.
Adoption is the user's decision. See
[the experiment contract](../../experiments/b200_last_layer_trim/README.md).

## Method

Causal LAST pooling reads only the final-token hidden state after layer 31,
a full-attention layer. `gleipnir.serving.last_query_attention` wraps only
that layer's FlashInfer backend forward. It leaves vLLM's KV-cache update
unchanged. It calls the same TRTLLM/MXFP8 prefill path with one query per
prefill request (each request's last scheduled token, bottom-right aligned
against the whole history), keeps decode-path rows, and zeroes the other
output rows. Tensor shapes, compilation, CUDA graphs and GEMM choices are
unchanged. Later layer-31 quantization is per row: FP8 per token, FROST FP4
row amax and SwiGLU whole-row amax. The worker requires causal LAST pooling, a
full-attention final decoder layer and the kernel receipt. Cascade attention,
non-TRTLLM prefill and fused output quantization raise errors.

## Exactness

The kernel gate passes all six cases bitwise (maximum difference 0): a 32K
first chunk, 32K chunks over 64K and 1M histories, a 17-token irregular query,
mixed prefill lengths and a 128-request batch.

In serving, all 65 timed synthetic scores and margins at every length, and
both phases' 20-row canaries, equal `million01` bitwise. The native canary
also reproduces the FP8 default exactly. The full320 c128 diagnostic matches
the baseline run's pattern against the six archived FP8 repeats: score MAE
0.0015, one threshold flip, and pooled/macro AUROC deltas of +0.010/+0.042 pp
(baseline run −0.025/+0.023 pp). Those differences come from c128 batch
composition, which varies between runs. They are not a trim effect, given the
bitwise C1 results.

## Speed

Same node, GPU UUID, driver and environment as `million01`; no drift was
detected. Startup took 72 s (native) and 70 s (YaRN), with warm caches.
Medians are over five requests.

| Context | Server | Baseline | Trim | Ratio | Saved | Fit prediction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1K | native | 35.47 ms | 35.65 ms | 1.005 | −0.2 ms | — |
| 2K | native | 35.04 ms | 35.72 ms | 1.019 | −0.7 ms | — |
| 4K | native | 37.93 ms | 37.37 ms | 0.985 | 0.6 ms | — |
| 8K | native | 55.82 ms | 52.70 ms | 0.944 | 3.1 ms | — |
| 16K | native | 98.66 ms | 97.86 ms | 0.992 | 0.8 ms | — |
| 32K | native | 207.49 ms | 203.35 ms | 0.980 | 4.1 ms | 9.0 ms |
| 64K | native | 484.47 ms | 470.88 ms | 0.972 | 13.6 ms | 26.9 ms |
| 128K | native | 1,256.5 ms | 1,198.9 ms | 0.954 | 57.6 ms | 89.8 ms |
| 256K | native | 3,648.0 ms | 3,413.1 ms | 0.936 | 234.9 ms | 323.1 ms |
| 32K | YaRN | 196.86 ms | 192.19 ms | 0.976 | 4.7 ms | 9.0 ms |
| 256K | YaRN | 3,542.95 ms | 3,306.88 ms | 0.933 | 236.1 ms | 323.1 ms |
| 512K | YaRN | 11.70 s | 10.68 s | 0.913 | 1,019 ms | 1,221 ms |
| 1M | YaRN | 43.08 s | 38.64 s | 0.897 | 4,437 ms | 4,739 ms |

The prediction removes one eighth of the fitted attention term,
71.8/8 · n(n+1)/2 ms over n 32K chunks. The realized share rises from 45% of
the prediction at 32K to 94% at 1M. One plausible explanation is that the
fitted attention term includes history-proportional work the trim keeps:
layer 31 still packs its whole K/V history to MXFP8 for each chunk. This is
not profiler-attributed. The 8K gain lies inside the baseline's 53.9–59.8 ms
repeat range, and the 1K/2K changes are within noise; short contexts gain
nothing measurable. Median SM clocks and peak power are unchanged within
sampling noise: long requests remain power-capped, and no extra clock benefit
appears.

The archived worker receipt records installation and the first (profiling)
call. Per-request trim activity is evidenced by the timings and bitwise
outputs, not by a running call count.

## Adoption

The user adopted `trim01` (last-query attention for all prefill requests, no
finished-only selection, no tail trim) into the FP8 serving default on
2026-10-15. `adopt02` runs a same-node A/B through the normal canary-guarded
startup: the untrimmed control first, then the adopted recipe. Both canaries
reproduce the default exactly (MAE 0). The adopted worker audits layer-31
installation.

| Workload | Control input tokens/s | Adopted | Ratio | Scores |
| --- | ---: | ---: | ---: | --- |
| quick64, c1, 3 passes | 94,149 (94,077–94,483) | 94,250 (94,172–94,322) | 1.001 | bitwise equal in every pass |
| full320, c128, 3 passes | 202,099 (185,798–208,629) | 213,650 (213,549–214,016) | 1.057 | within batch nondeterminism |

At c128, the adopted minimum exceeds the control maximum; median p50 latency
falls from 2.37 to 2.28 s. Full320 contains prompts up to about 28.7K tokens,
so attention is a larger share than in quick64. c128 scores vary with batch
composition: control-vs-control passes differ by MAE 0.0035 (maximum 0.286),
and control-vs-adopted by the same MAE 0.0035. Three passes per arm do not
establish a precise batched gain.

## Artifacts

`results/b200_last_layer_trim/trim01/` (and `trim02`/`trim03`) contain the kernel receipt
(`last_query.json`), per-phase commands, canaries, audits, quality diagnostic,
raw repeats, GPU samples, environment snapshots and `comparison.json` (from
`experiments/b200_last_layer_trim/compare.py`). Logs are under
`logs/runpod/b200_last_layer_trim/`. Both servers were retired by identity
check. The GPU is idle and the Pod remains running.

## Follow-ups: finished-only attention and tail trim

Both follow-ups were user-requested and frozen before running, on the same
node with no environment drift.

**`trim02`, finished-only attention: exact, no extra gain.** Layer-31
attention, and its MXFP8 history packing, run only for prefill requests whose
prompt completes in the step. The kernel gate passes 8/8 bitwise, including
two request-subset cases. The audit shows 1,354 of 1,575 native and 44 of 362
YaRN prefill steps attended. All synthetic and canary scores equal `million01`
bitwise. Latency matches `trim01` within noise: 32K 206.6 ms, 256K 3.43 s,
512K 10.73 s, 1M 38.56 s (−10.5% vs baseline). Packing the history on
mid-prompt chunks was therefore not a measurable cost.

The shortfall against the fitted b/8 prediction at shorter lengths has
another cause. One plausible cause is that the median SM clock falls with
context, so even per-chunk linear work slows at long lengths, and the fit
assigns that to the attention term. This is not profiled.

**`trim03`, finished-only attention plus the layer-31 tail on selected rows:
not exact, no gain.** The gate, output projection, post-attention norm and MLP
run eagerly on pooled rows inside a vLLM splitting op. Latency equals `trim02`
within noise at every length (1K 35.48 ms, 256K 3.40 s, 1M 38.32 s). Scores are
no longer bitwise. Synthetic scores differ from `million01` by up to 0.037
(typically 0.006–0.016), and the native canary MAE is 0.0040 against the FP8
default, inside the 0.005 limit. The full320 c128 diagnostic gives score MAE
0.0078 against the archived repeats (0.0011–0.0015 for the other runs), two
threshold flips, and pooled/macro AUROC +0.039/−0.58 pp.

This drift exceeds low-bit rounding. A plausible cause is that the eager
small-row MLP takes a different FP4/SwiGLU quantization path from the compiled
graph; this is unverified. With no measurable speed benefit, the tail trim is
not worth its numerical change.

**Assessment.** The single-request saving comes from removing layer-31
attention (`trim01`). Finished-only selection is equally exact and equally
fast, but depends on runner state for no measured gain. The simplest exact
variant, `trim01`, is the natural candidate if adopted. Adoption remains the
user's decision.
