# Lean GDN prefill host path

The [host-overhead probe](../b200_host_overhead/README.md) attributed about
12.7 ms of each short C1 step's ~32 ms host time to vLLM's GDN core across 24
layers. User-requested 2026-10-15: streamline that path without changing
kernels or numerics.

`gleipnir.serving.gdn_fast` replaces each GDN layer's
`_forward_core_fused_norm_packed` for prefill-only, non-speculative FlashInfer
batches. It calls the same conv1d, post-conv, FlashInfer delta-rule and gated
norm kernels with the same inputs, and:

- computes int64 `cu_seqlens` and the "every request starts from empty state"
  flag once per forward rather than per layer;
- has FlashInfer write straight into `core_attn_out`, with no copy;
- for fresh requests, starts FlashInfer from zero state and writes final states
  into the pool via `state_indices`, instead of gather, zero-mask and scatter;
- skips vLLM's CustomOp and FlashInfer wrapper layers.

Other batches use the original method. `FastGdnWorker` installs it on the
adopted gated worker, with `set_gdn_fast` and `gdn_fast_audit` RPCs.

## Validation and results (EU-RO-1 B200)

- `native.py`: FlashInfer's zero-state pool path equals the gathered zero-state
  path bitwise (output, state, untouched other slots) on non-CP and CP, for
  1–4 sequences and 1K–32K tokens (`native02`).
- `paired.py` (`fast01`): in-server toggle, four alternating rounds. C1 scores
  are bitwise equal in every case.

  | Case | Upstream p50 | Fast p50 | Paired rounds faster |
  | --- | ---: | ---: | --- |
  | 1K | 32.40 ms | 29.48 ms | 4/4 |
  | 4K | 34.09 ms | 31.35 ms | 3/4 |
  | quick64 | 32.66 ms | 29.73 ms | 4/4 |
  | 8K | 50.49 ms | 48.56 ms | 3/4 |
  | 32K | 196.83 ms | 192.86 ms | 4/4 |

  full320 c128 throughput ratios: 1.017, 1.016, 0.975, 1.020.
- `canary_check.py` (`canary01`): the c4 canary is repeatable within a server
  and bitwise equal across arms. The startup canary's larger MAE (0.00125,
  versus 0.0005 in other servers) reflects how that instance batched its first
  c4 pass, not the fast path. c4 and c1 canary scores differ by up to 0.053
  from batch composition alone.

```bash
... -m experiments.b200_gdn_fast.native --output results/b200_gdn_fast/native02.json
... -m experiments.b200_gdn_fast.paired --name fast01
... -m experiments.b200_gdn_fast.canary_check --name canary01
```

## Strided gated norm (exact, adopted)

Upstream's gated RMS norm flattens the output gate, a strided slice of the
packed projection, to `[tokens × heads, 128]`. That forces an `aten::reshape`
clone, which is most of the profile's direct-copy time: 7.3 ms per 32K step,
and 292 ms per full320 c128 pass. `strided_gated_norm` launches the same FLA
Triton kernel over `[tokens, heads × 128]` rows with `group_size=128`, the
per-head weight tiled across groups, and the `ROWS_PER_BLOCK` upstream picks
for `tokens × heads` rows. Each (token, head) row therefore reduces with the
same tile shape. Nothing is copied.

- `norm_probe.py` (`norm_probe01`): bitwise equal to upstream for 1–32,768
  tokens and every norm-before-gate/activation combination (24/24).
- `paired_norm.py` (`norm01`): one server with the adopted stack, four
  alternating rounds. Every C1 score is bitwise equal.

  | Case | Copy | Strided |
  | --- | ---: | ---: |
  | 8K | 48.9 ms | 46.8 ms |
  | 16K | 95.1 ms | 88.6 ms |
  | 32K | 197.4 ms | 189.2 ms |
  | quick64 | 30.4 ms | 29.7 ms |

  full320 c128 throughput ratios are 1.054/0.975/1.049/1.041.

The user adopted it on 2026-10-15. It is on by default, startup requires
that it ran, and the selection binds the file and both receipts.
