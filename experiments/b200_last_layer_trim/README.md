# Last-query attention in the final layer

With causal LAST pooling, the score reads only each request's final-token
hidden state after layer 31, which is a full-attention layer. Layer 31 must
still write K/V for every token, because later chunks and the final token
attend to them. Its attention output for every other token is never pooled.
This intervention keeps vLLM's KV-cache update unchanged and runs layer-31
attention for one query per prefill request: the request's last scheduled
token, at bottom-right causal alignment against its whole history. Decode-path
rows are unchanged. Other output rows are zero.

Every tensor keeps its shape, so compilation, CUDA graphs and GEMM kernel
choices are unchanged. All later layer-31 quantization is per row: FP8
per-token scales, FROST FP4 row amax, and SwiGLU whole-row amax. So the
selected rows should be bitwise identical to the baseline. The output
projection and MLP still run on all rows; trimming them is a separate,
smaller follow-up (about 1/32 of linear work).

Reusable code: `src/gleipnir/serving/last_query_attention.py`. The worker
installs it on the final attention layer only after checking causal LAST
pooling, that the final decoder layer is full attention, and the kernel
receipt. Unsupported paths (cascade, non-TRTLLM prefill, fused output
quantization) raise rather than fall back.

## Hypothesis and predictions

The [1M baseline](../../docs/findings/b200_context_million.md) fit gives
b = 71.8 ms per 32K×32K attention block across 8 layers. Removing one layer's
attention saves about b/8 · n(n+1)/2 over n 32K chunks: 9 ms at 32K (4.3%),
0.32 s at 256K (8.9%), 1.22 s at 512K (10.4%) and 4.74 s at 1M (11.0%). Lower
power draw may add a clock benefit on this power-capped node. Below 32K the
saving is proportionally smaller.

## Frozen protocol and acceptance

Use the `b200_context_million` runner and settings unchanged, on the same
node, with only the worker and bound sources changed. The control is the
completed `results/b200_context_million/million01`; do not rerun it.

1. Kernel gate (`native.py`, before any server starts): the MXFP8 output of
   last-query attention must equal the full-chunk rows bitwise for six cases.
   These are a 32K first chunk, a 32K chunk over 64K and over 1M histories, an
   irregular 17-token query, mixed prefill lengths, and a 128-request batch.
   Any mismatch stops the run.
2. Serving exactness: native-phase canary scores reproduce the FP8 default.
   Synthetic scores at every length in both phases, and both phases' canary
   scores, are compared with `million01`. The expected result is bitwise
   equality. Any difference is reported with its maximum absolute size and
   blocks a "no numerical change" claim. It does not stop timing.
3. Quality diagnostic: three full320 c128 passes against the six archived FP8
   default repeats, as in the baseline. This is a short-input diagnostic, not
   long-context quality.
4. Speed: median C1 latency and input tokens/s per length against `million01`,
   with clock/power samples. There is no promotion; the user decides adoption.

## Follow-ups: finished-only attention and tail trim

User-requested after `trim01`. Both reuse the same runner, settings, node and
`million01` control, and are frozen before running:

- `finished` (`trim02`): layer-31 attention runs only for prefill requests whose
  prompt completes in this step. This is vLLM's own pooling test,
  prompt length == computed + scheduled tokens, read from the runner's CPU state
  in attention-batch order and checked against the attention metadata. A
  mid-prompt chunk does no layer-31 attention or MXFP8 history packing. The
  kernel gate adds subset cases (every other request) through the serving
  metadata-selection helper. Expected: bitwise equality with `million01`,
  as for `trim01`; speed closer to the fitted b/8 prediction.
- `tail` (`trim03`): additionally, layer 31's gate, output projection,
  post-attention norm and MLP run eagerly only on the selected rows. This is a
  custom op registered as a vLLM splitting op (like attention), so CUDA graphs
  never capture a row selection. Small-M GEMM kernels and eager-versus-compiled
  elementwise rounding may change low bits. Required: canary reproduction of
  the FP8 default (MAE <=0.005, correlation >=0.995), finite outputs, and
  reported synthetic score differences from `million01`. Expected gain: about
  1/32 of linear work, under 2.5% at short contexts and under 0.5% at 1M.

```bash
... -m experiments.b200_last_layer_trim.run --name trim02 --variant finished
... -m experiments.b200_last_layer_trim.run --name trim03 --variant tail
```

Outcome: `trim02` is bitwise exact with no gain over `trim01`; `trim03` is
not exact (scores up to 0.037 from the baseline) and no faster. See the finding.

Stop on failed kernel gate, identity/canary/token/finite failure, OOM, timeout
or completion. Retire only this run's verified server; never change capacity.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_last_layer_trim.run --name trim01
```

Results go under `results/b200_last_layer_trim/<name>/` and logs under
`logs/runpod/b200_last_layer_trim/`. Compare with the baseline using
`python -m experiments.b200_last_layer_trim.compare
results/b200_last_layer_trim/trim01`. The completed `trim01` run is bitwise
unchanged and 10.3% faster at 1M; see
[the finding](../../docs/findings/b200_last_layer_trim.md).
