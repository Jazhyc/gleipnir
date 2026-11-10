# Serving ladder for Gleipnir 4B

User request, 2026-11-10: a matched measurement of the throughput each
serving step adds for the 4B, for the cost table of the v1 report. The
[27B ladder](../qwen38_27b_serving/README.md) already exists on the same
benchmark; the 4B has only pairwise comparisons from separate sessions and no
stock BF16 baseline on a B200.

- **Question.** How much prompt throughput does each cumulative serving step
  add for Gleipnir 4B on one B200?
- **Rungs** (cumulative; each is the selected recipe's own worker layer):
  1. `bf16`: stock-arithmetic BF16 vLLM 0.31 (`Bf16Worker`), prefill only.
  2. `quant`: FP8 attention projections, FP4 MLP and GDN projections, MXFP8
     attention (`AttentionPrecisionWorker`).
  3. `lastquery`: plus last-query final-layer attention.
  4. `route`: plus the length-gated GDN route.
  5. `fast`: plus the lean GDN host path. This is the released recipe at its
     recorded 32K context; `run.py` requires its command to equal the
     release command.
- **Model.** The released merged checkpoint `Jazhyc/Gleipnir-4B-BF16@v0.4`
  for every rung, so only the serving path changes.
- **Workload and harness.** As for the 27B: frozen `full320` at concurrency
  128 (three repeats after a `quick64` warmup) and `quick64` at concurrency
  1, through `/v1/monitor/score`, prefix caching off. Prompt tokens/s is the
  main measure.
- **Projection.** `compare.py` converts median tokens/s to USD per 1,000
  evaluations at 8,654.7 prompt tokens per evaluation (the 4B's
  compact-instruction OOD workload) and $7.99/h and $3.44/h, assuming full
  utilization.
- **Scores** are saved for a sanity check only (MAE against the `fast`
  rung). Quality on held-out data is in `experiments/qad_v02/README.md`.
- **Stop** on a failed startup, a command that differs from the release
  command, a failed BF16 audit, a non-finite score or completion.
- **Compute.** One B200 Pod from the release image
  `ghcr.io/jazhyc/gleipnir-serve:0.2`, authorized by the user on 2026-11-10.
  EU-RO-1 (where the 27B ladder ran) had no capacity, so the user approved
  another datacenter; the Pod ran in US-CA-2.

```bash
# On the Pod (release image), per rung:
PYTHONPATH=src:. $PY -m experiments.gleipnir4b_serving_ladder.run --rung quant
# Anywhere with the results:
PYTHONPATH=src:. python -m experiments.gleipnir4b_serving_ladder.compare
```

## Result

Run 2026-11-10, 18:36–18:52 UTC, Pod `x82ld1enklasda` (US-CA-2, B200
`GPU-10074f28…`, driver 595.91.07, EPYC 9555; deleted after the artifacts
were pulled and checksum-verified). The fully layered command equals the
release command. Two sessions, a fresh server per rung: session 1 with three
c128 repeats, session 2 (`*_r2`) with six. Session 2 is the reported one: in
session 1 the `quant` rung was still warming up (177K, 199K, 210K) and
`route` had one slow pass. Medians of session 2
(`results/gleipnir4b_serving_ladder/comparison.json`):

| Rung (cumulative) | c128 prompt tok/s, median (range) | vs BF16 | vs previous | Projected $/1K at $7.99/h | at $3.44/h |
|---|---:|---:|---:|---:|---:|
| `bf16` | 118,933 (115,463–119,299) | 1.00× | | 0.1615 | 0.0695 |
| `quant` | 210,609 (200,994–211,430) | 1.77× | 1.771× | 0.0912 | 0.0393 |
| `lastquery` | 212,889 (202,432–214,338) | 1.79× | 1.011× | 0.0902 | 0.0388 |
| `route` | 213,398 (133,162–214,460) | 1.79× | 1.002× | 0.0900 | 0.0388 |
| `fast` (released) | 225,964 (212,721–226,657) | 1.90× | 1.059× | 0.0850 | 0.0366 |

- Low-precision arithmetic gives most of the gain (1.77×). Last-query
  attention, the route gate and the lean host path together add 1.073×.
- Session 1 medians agree for the stable rungs (118,790; 213,184; 226,051).
- In every session-2 rung the fifth pass is the slowest (3–6% lower; 38%
  lower for `route`), so the median is used and the cause is not diagnosed.
- Last-query attention adds 1.1% here against 5.7% in the
  [adoption A/B](../../docs/findings/b200_last_layer_trim.md), which ran on
  a different host and stack state.
- The projected $0.0850 for the released recipe equals the cost recorded
  from the measured OOD wall time (244.9 s for 55,346,879 prompt tokens,
  226K tokens/s;
  [frontier record](../../docs/research/tool_trajectory_ood_frontier.md)),
  so the benchmark and the OOD suite agree at the end of the ladder.
- Score sanity: mean absolute difference from `fast` is 0.003–0.005 for the
  optimized rungs and 0.049 for `bf16` on this training-seen cohort.
