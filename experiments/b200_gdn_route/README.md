# Length-gated GDN prefill route

FlashInfer's `use_cp="auto"` selects its single-GPU context-parallel (CP) GDN
prefill from batch-head parallelism alone. A single request therefore takes CP
even at 1K tokens. The [host-overhead probe](../b200_host_overhead/README.md)
found that CP off saves about 4 ms per 1K step (C1 p50 36.1 → 31.2 ms), with
canary MAE 0.0005 against the default. CP should still help long single
chunks. User-requested 2026-10-15: gate the route by length and benchmark it.

`gleipnir.serving.gdn_route` forces `use_cp=False` when a FlashInfer GDN call
has fewer than `cp_min_tokens` scheduled tokens. Otherwise it leaves
FlashInfer's `"auto"` decision untouched, so long chunks and large batches keep
today's exact arithmetic. It composes only with `GLEIPNIR_FLASHINFER_GDN_CP=auto`
and pins the FlashInfer version and source.

## Frozen protocol

All runs are on the same EU-RO-1 node, with the native-RoPE C1 sweep (1K–256K)
and settings of the [1M baseline](../b200_context_million/README.md), on the
adopted last-query worker. The CP-auto control is the completed
`results/b200_last_layer_trim/trim01` (same worker and settings); do not rerun
it.

1. `off01`: CP off for every call. Canary, quality diagnostic and timings as
   in the sweep.
2. **Threshold rule.** `cp_min_tokens` is the smallest power of two L in
   1K–32K at which `trim01`'s median latency is at most `off01`'s, at L and at
   every larger power of two up to 32K. If CP never wins, the threshold is
   32,769, meaning non-CP for every chunk. Chunks never exceed 32K, so this
   covers every per-call decision.
3. `gated01`: the selected threshold. Expected: scores bitwise equal to
   `trim01` at lengths at or above the threshold, and to `off01` below it;
   latency matching the faster route at each length; canary within the
   default's reproduction limits.
4. If the gate is adopted, run a same-node quick64 c1 / full320 c128 A/B against
   the adopted default through the normal startup.

```bash
... -m experiments.b200_gdn_route.run --name off01 --route off
... -m experiments.b200_gdn_route.run --name gated01 --route gated --cp-min-tokens N
```

## Outcome

`off01` and `gated02` (`gated01` aborted on the idle-GPU guard) plus the
in-server paired A/Bs `paired01` (1K/4K/quick64) and `paired02` (8K–32K)
confirm `cp_min_tokens = 8192`. Run the paired test with
`python -m experiments.b200_gdn_route.paired --name <n> [--cases crossover]`.
See [the finding](../../docs/findings/b200_gdn_route.md).
