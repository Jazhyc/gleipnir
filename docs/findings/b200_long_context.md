# B200 single-request long contexts

2026-10-07 UTC, recorded 2026-10-08 locally. Exact 262,144-token inputs complete
at concurrency 1 without OOM after extending the attention envelope and fixing
the pinned pooling scheduler's final-token reservation. Median warmed latency
is 3.60 seconds and input throughput 72,795 tokens/s. The original selected
server's 32K limit and the failed first 256K attempt remain recorded separately.
See [the experiment contract](../../experiments/b200_long_context/README.md).

## Contract and serving changes

Use deterministic unlabelled synthetic text at 8,192/16,384/32,768/65,536/
131,072/262,144 tokens. Freeze decoded prompt hashes, generator/tokenizer hashes
and exact token counts. At each length exclude one warmup, then measure five
sequential requests with fresh HTTP pools. Timings include localhost HTTP,
native text encoding, engine waiting and inference. No input truncation,
prefix-cache reuse, generated output tokens, held-out evaluation or promotion.

Preserve the selected merged model/two-row head, native Gigatoken/direct FROST,
FP4 MLP/GDN/attention projections, MXFP8 full-attention prefill, BF16 cache and
operands, FP32 gates/state, 32,768-token chunk budget, 128 engine-sequence cap and
FCFS order. Client concurrency is one throughout. The checkpoint's native
position limit is 262,144. Raise the serving context limit and opt-in MXFP8
descriptor/producer guard to that limit; keep original bound serving sources
unchanged. This is an experimental capacity extension, not a baseline selection.

The original process first measures 8K/16K/32K at median 57.53/109.22/209.16 ms
and 142,275/149,938/156,626 input tokens/s, then returns HTTP 400 on exact 64K.
Preserve that control and explicit configured-limit rejection in `long01`.
The independent extended native check matches original short and 128-sequence
outputs bitwise. Full 32K query chunks against 64K/128K/256K histories and an
irregular 32K+17 history remain finite; sampled independent quantized-arithmetic
relative L2 is 0.001644–0.001667. Strict BF16-reference relative L2 is
0.05190–0.05414, failing the 0.05 diagnostic gate in all four cases. Preserve
those failures separately from arithmetic agreement and the inherited accepted
FP4/MXFP8 finite-quality tradeoff; no new precision claim follows.

## Exact-cap scheduler failure and correction

The first extended worker serves through 128K, then stalls on exact 256K with
one running request, zero subsequent GPU work and no OOM. In pinned vLLM 0.24.0,
`Scheduler.num_sampled_tokens_per_step` defaults to one for non-diffusion models,
including pooling. Its running-prefill clamp reserves that position at the
context cap; the waiting path lets one-chunk requests complete, explaining why
the original exact 32K request succeeded. A test using the installed scheduler
and CPU KV allocator reproduces progress to `max_model_len - 1`, no further
scheduled tokens, then final-token progress after correction.

`PoolingContextScheduler` sets this reservation to zero only for pooling and
requires pinned upstream/integration source hashes, synchronous scheduling and
FCFS. It changes no queue order, GPU arithmetic or token budget. Stop the
identity-verified stalled worker, retain its failure/source/GPU receipt and
retry all six lengths as `long02` with this correction. The first extended
configuration takes 272.70 s to readiness; the corrected retry reuses its
`b908978039` compiled cache and takes 100.42 s. Both startup times are excluded
from the warmed measurements below.

## Completed measurements

All rows use the same corrected extended worker. K means 1,024 tokens.
Memory is peak PyTorch allocated memory, including persistent caches; reserved
allocator memory and NVML process memory are distinct counters.

| Exact context | Median latency | Input tokens/s | Requests/s | Peak allocated |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 51.50 ms | 158,915 | 19.399 | 158.92 GiB |
| 16K | 98.23 ms | 166,689 | 10.174 | 159.68 GiB |
| 32K | 200.90 ms | 163,065 | 4.976 | 161.21 GiB |
| 64K | 469.93 ms | 139,437 | 2.128 | 161.21 GiB |
| 128K | 1231.21 ms | 106,451 | 0.812 | 161.21 GiB |
| 256K | 3601.03 ms | 72,795 | 0.278 | 161.21 GiB |

Five-repeat latency ranges in length order: 50.53–61.13, 97.90–103.04,
200.39–202.22, 464.12–470.87, 1223.74–1238.54 and 3588.94–3634.17 ms.
Input-throughput ranges are 133,835–161,971; 158,894–167,238;
161,988–163,479; 139,158–141,183; 105,821–107,101 and 72,131–73,040 tokens/s.
No precise descriptor/restart speed gain is inferred from sequential short
controls. From 32K to 256K, length grows eight-fold and latency 17.92-fold.
This is consistent with increasing full-attention/history-packing cost; it is
not a profiler-based attribution of that increase.

Peak incremental PyTorch allocation is 0.766/1.531/3.063 GiB at 8K/16K/32K,
then remains 3.063 GiB through 256K. Peak reserved memory is 163.78 GiB from
32K upward. Chunking bounds the largest temporary query workload and cache
storage is already allocated. Worker telemetry observes 32K query chunks with
histories 32K, 64K, 96K, 128K, 160K, 192K, 224K and 256K. This demonstrates
full input processing at C1; it does not establish multi-request 256K capacity.

## Numerical diagnostics and closure

Real 20-row adapter canaries pass before original timing, after extension and
after boundary correction: mean score error 0.000807, correlation 0.999967,
nonzero adapter effect 0.839906. Every synthetic response has exact input usage
and finite consistent scores/margins. Long-context model quality is unmeasured.

Three full320 c128 passes on the first extended worker are paired with all six
archived selected quality controls: pooled AUROC delta −0.22269 percentage
points, source-macro −0.13741 pp and zero median threshold flips. Per-source
deltas, undefined single-label sources, calibration, ties and repeat variation
remain in `quality.json`. These short-input batch diagnostics are explicitly
reused in the retry because its correction affects only the unused context-cap
reservation; native/kernel/runtime/source identities remain unchanged. They
do not isolate a quality effect or promote the extension. Fresh stock-reference
batch drift is already recorded in [the scaling finding](b200_score_scaling.md).

All 81 failed-attempt and 70 completed-attempt remote artifacts match collected
checksums. Results, prompt/source/CPU bindings, raw repeated measurements,
native checks and both server logs are in `results/b200_long_context/long01/`
and `long02/`. The latter exports source-bound CSV/JSON and PNG/SVG figures.
Seven tests pass, including the real upstream boundary reproduction; Ruff is
clean. Local tests retain existing NVML and TorchScript deprecation warnings.
The input-construction tokenizer warning concerns pre-crop text; endpoint
counts verify that only the exact requested lengths reach the model.

At benchmark closure, health is 200 and API/engine 29162/29185 are kept warm;
NVML reports 168774 MiB allocated out of 183359 MiB and 0% utilization.
On 2026-10-08 the user explicitly ends serving optimization and requests stopping
vLLM. Identity-checked retirement verifies both processes exited, no GPU
processes and 0 MiB GPU use. `long02/user_stop.json`, `user_stop_server.log` and
their separate checksum manifest preserve this closure. The Pod remains running;
shared caches, checkpoints, benchmark results and failed receipts remain intact.
No capacity lifecycle operation or automatic reference restore occurs.
