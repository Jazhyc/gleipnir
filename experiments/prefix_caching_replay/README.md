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
- **Measures.** Per check: latency, prompt tokens, score. Per trajectory:
  prefix-cache queried and hit tokens, read from `/metrics` before and after
  its checks (amended 2026-11-20 before the EU run: the US run read them around
  every check, which may add latency; `probe.py` measures that cost). Per trajectory: total check time cached and uncached, and
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

## Attempts

- **2026-11-19, `5ndq8wcpbn8bmv` (US-TX-6, image 1.1).** Restart-looped: the
  start command `/start.sh` reached the release entrypoint as a model name,
  and Runpod's REST API, which earlier Pods used to override the entrypoint,
  no longer resolves (GraphQL cannot override it). Deleted after about 20
  minutes with nothing to collect; this led to image 1.2.
- **2026-11-19/20, `ypx6hooyaa38zi` (US-NC-2, image `1.2-rc1`, B200
  `GPU-ca5da4f7…`, Xeon Platinum 8568Y+, driver 595.112.07).** Completed the
  image 1.2 check (see the release README) and three of the four replay
  passes (`results/prefix_caching_replay/replay01_us_nc2/`, logs alongside).
  Not used for the result: the user asked for EU-RO-1, the host of the
  report's serving numbers. Per-request latency was far above those
  numbers: isolated checks ran at 89.5K prompt tokens/s for the 4B (205K
  on EU-RO-1) and 39.6K for the 27B (50.8K), with a floor of about 70 ms
  even for checks of a few hundred new tokens. On this host the 4B's
  per-action time fell only from 214 s uncached to 193 s cached, although
  88% of prompt tokens hit the cache. Whether the floor comes from the host
  or from the replay's `/metrics` scrapes is what `probe.py` separates.

