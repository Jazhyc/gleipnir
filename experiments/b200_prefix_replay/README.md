# Prefix-cached replay of a growing monitored session

This tests how much vLLM prefix caching helps synchronous monitoring, where the
monitor rescores a growing agent transcript before each action.

**Workload** (`prepare.py`, `data/b200_prefix_replay/`): one synthetic session.
It takes the first Gloom trajectory's header, then chains agent steps from 62
consecutive Gloom trajectories (strict-filtered training rows) up to
1,037,565 tokens at the final check: 611 steps.
- A check sends the monitor prompt with the transcript through that step's
  proposed tool call.
- Consecutive checks share an exact text prefix.
- It is an unlabelled systems workload, so its scores carry no quality meaning.

**Policies:**
- `every_action`: 598 checks.
- `escalated`: 52 checks, a seeded 7.2% sample matching OpenAI's Auto-review
  snapshot (720 of 10,000 actions reviewed).

**Server.** The adopted full stack (last-query trim, 8K GDN gate, lean GDN
path, strided norm) on the 1M YaRN long-context server. The only command
changes are `--enable-prefix-caching --mamba-cache-mode align`. vLLM warns
that it does not officially support prefix caching for hybrid pooling models,
so score parity is required.

**Measurements** (concurrency 1, sequential):
- latency, prompt tokens, and prefix-cache queried/hit tokens (from
  `/metrics`) per check;
- each policy starts after `/reset_prefix_cache`;
- an uncached pass rescores about 8 checks at 16K–1M after a cache reset each,
  giving uncached latency and cached-versus-uncached score differences.

```bash
PYTHONPATH=src:. python -m experiments.b200_prefix_replay.prepare
PYTHONPATH=src:. <serving-python> -m experiments.b200_vllm031.runtime \
  -m experiments.b200_prefix_replay.run --name replay01
```

## Result (`replay03`, 2026-10-16, EU-RO-1 B200)

`replay01` failed because the container restart had removed the ephemeral
merge; `merge03` rebuilt it with identical checksums. `replay02` failed
because the production monitor audit rejects prefix caching; `worker.py`
masks the flag only during that audit. Both failures are retained.

| Policy | Checks | Cached total | Uncached estimate | Speed-up | Prefix hit fraction | Recomputed tokens/check (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Every action | 598 | 595 s | 7,665 s | 12.9× | 99.6% | 1,836 |
| Escalated (7.2%) | 52 | 95 s | 747 s | 7.9× | 96.2% | 14,432 |

The uncached estimates interpolate the same-server uncached rescoring
(log-log). Median per-check latency for every-action monitoring:

| Context | Cached | Uncached |
| --- | ---: | ---: |
| <64K | 0.09 s | 0.22 s |
| 64–256K | 0.32 s | 1.84 s |
| 256–600K | 0.79 s | 7.1 s |
| 600K–1.04M | 1.54 s | 23.8 s |

At 1.04M tokens: 2.15 s cached versus 38.0 s uncached. Cached latency still
grows with context, because the recomputed tokens attend over the whole
history.

**Score parity: not exact.** Rescoring 8 checks (15K–1.04M) uncached gives
|cached − uncached| of 0.054, 0.034, 0.028, 0.062, 0.060, 0.151, 0.048 and 0.0
(mean 0.055). That matches the deterministic path-dependence measured
elsewhere: the GDN route gate changed C1 scores by MAE 0.055. Cached requests
process about 2K new tokens per call (non-CP GDN route, different chunk
boundaries), whereas uncached prefill runs 32K chunks (CP route). Enabling
prefix caching also shifts the c4 canary slightly against the same YaRN stack
without caching (MAE 0.0043). The cause is not isolated. Before production use,
compare cached and uncached with the GDN route held fixed, and decide whether
this path-dependence is acceptable.

## Opt-in prefix-cached serving

`python -m experiments.b200_attention_precision.startup --name NAME --prefix-caching`
launches the selected default with align-mode prefix caching. The only changes
are `--enable-prefix-caching --mamba-cache-mode align` and the
`PrefixCachedFastGdnWorker` subclass. All startup audits and the canary gate
still apply. Use it for synchronous monitoring of growing sessions only:
- cached scores differ from uncached scores by up to about 0.15;
- benchmark and evaluation runners replay identical prompts and must keep
  caching off (they assert `--no-enable-prefix-caching`).
