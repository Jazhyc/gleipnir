# Per-action monitoring with prefix caching on the released stack

User request, 2026-11-19 (authorized: one B200 Pod, deleted after the run):
measure what monitoring every action costs for the released Gleipnir 4B and
27B, with and without prefix caching, on the report's serving benchmark.
pplx-decider is not run: its bidirectional attention rules out caching, and
its per-action cost follows from its billed tokens
([token count](../prefix_caching_tokens/README.md)).

## Design (frozen before the Pod)

- **Hypothesis.** With prefix caching, checking every action costs a small
  multiple of one isolated check; without it, a multiple close to the
  token ratio (11.2 on this workload).
- **Workload.** The 96 agent trajectories of the serving benchmark
  (`data/b200_inference_benchmark/full.json`, frozen hash), in their order.
  One check per tool-calling assistant turn (1,882 checks, 5–78 per
  trajectory): the prompt through that turn plus the prompt's closing suffix.
  After its checks, each trajectory's full prompt is sent once as the
  isolated check.
- **Serving.** `gleipnir-serve:1.1` on one B200, its own entrypoint at the
  default 262K context, one request at a time. Per model: the default mode
  (`uncached`), then a restart with `--prefix-caching` (`cached`, align mode;
  the cache is reset before each trajectory). One server at a time; three
  untimed warmup checks per server.
- **Measures.** Per check: latency, prompt tokens, score, prefix-cache queried
  and hit tokens. Per trajectory: total check time cached and uncached, and
  the ratio to the uncached isolated check. Cost per 1,000 trajectories at
  $7.99/h and $3.44/h. Score drift: |cached − uncached| per check.
- **Primary result.** Cached and uncached per-action time over the 96
  trajectories, as multiples of the isolated time, per model.
- **No selection.** One pass per model and mode; nothing is tuned. The
  workload is training-seen, so scores are not a quality measurement.
- **Stop conditions.** A server that fails to start or a nonfinite score
  stops that model. Infrastructure failures resume with the same command;
  each trajectory is scored at most once per model and mode.

```bash
# on the Pod, after pushing experiments/prefix_caching_{tokens,replay} and data/b200_inference_benchmark
experiments/prefix_caching_replay/pod_run.sh replay01 4b 27b
```
