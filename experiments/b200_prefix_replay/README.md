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
