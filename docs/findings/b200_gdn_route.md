# Length-gated GDN prefill route

2026-10-15, on the EU-RO-1 B200 with the adopted last-query default. FlashInfer's
single-GPU context-parallel (CP) GDN prefill is slower than its non-CP route
for chunks under 8K tokens and faster from 8K up. A gate that forces non-CP
below 8,192 tokens per call saves about 2.3–2.9 ms (7–8%) on short C1
requests. Long chunks and batches keep their exact current arithmetic. Not yet
adopted. See [the experiment](../../experiments/b200_gdn_route/README.md).

## Why

FlashInfer's `use_cp="auto"` picks CP from `num_seqs × heads` parallelism
alone, so a single request always takes CP. CP adds host-side planning plus
four kernels per layer (two precompute passes, prefill, fix-up). That pays
off only when the chunk is long enough to need sequence-level parallelism on
one GPU. It is unrelated to vLLM's multi-GPU context parallelism.

## Evidence

The sweeps (`off01`: CP off everywhere; `gated02`: gate at 8,192) use the same
worker and settings as `trim01`. Server instances differ by about 2 ms at
short lengths. The `gated02` short-length timings look no better than CP auto,
yet its scores there are bitwise equal to `off01`, so it ran non-CP; the
difference is instance noise. Route effects were therefore measured within
one server, switching the gate threshold at runtime in six alternating rounds
per case (`paired01`, `paired02`):

| C1 case | CP auto p50 | Non-CP p50 | Paired rounds won |
| --- | ---: | ---: | --- |
| 1K synthetic | 33.5 ms | 31.2 ms | non-CP 6/6 |
| 4K synthetic | 35.3 ms | 33.0 ms | non-CP 6/6 |
| quick64 (real) | 34.5 ms | 31.6 ms | non-CP 6/6 |
| 8K synthetic | 50.8 ms | 51.7 ms | CP 6/6 |
| 16K synthetic | 94.1 ms | 96.9 ms | CP 6/6 |
| 32K synthetic | 195.4 ms | 199.7 ms | CP 6/6 |

This confirms the frozen-rule threshold of 8,192, which `off01` against
`trim01` had suggested, with a cross-instance margin at 8K of only 1.9 ms.
Scores were stable within each arm.

**Numerics.** At and above the threshold, the gate leaves FlashInfer's own
choice in place, so outputs are unchanged. Below it, outputs are the non-CP
route's: the canary differs from the default by MAE 0.0005 (correlation
0.999994), inside the 0.005 reproduction limit. The full320 c128 diagnostic
against the archived FP8 repeats gives score MAE 0.0012, one threshold flip,
and pooled/macro AUROC −0.10/−0.02 pp, comparable to the other runs' batch
variation. At 32K and above, `gated02` C1 scores equal `million01` bitwise
wherever both exist.

## Notes

- `gated01` aborted on the idle-GPU guard because it was launched before
  `off01` finished; it is retained as an aborted record.
- The 1K timings in the [host-overhead probe](../../experiments/b200_host_overhead/README.md)
  (36.1 vs 31.2 ms) compared separate server instances. The paired 2.3 ms is
  the attributable route effect.
