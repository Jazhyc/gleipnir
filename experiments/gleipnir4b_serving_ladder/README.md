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
  evaluations at 9,466.7 prompt tokens per evaluation (the OOD mean) and
  $7.99/h and $3.44/h, assuming full utilization. The benchmark cohort is
  shorter than the OOD suite, so these are projections.
- **Scores** are saved for a sanity check only (MAE against the `fast`
  rung). Quality on held-out data is in `experiments/qad_v02/README.md`.
- **Stop** on a failed startup, a command that differs from the release
  command, a failed BF16 audit, a non-finite score or completion.
- **Compute.** One EU-RO-1 B200 Pod from the release image
  `ghcr.io/jazhyc/gleipnir-serve:0.2`, authorized by the user on 2026-11-10.

```bash
# On the Pod (release image), per rung:
PYTHONPATH=src:. $PY -m experiments.gleipnir4b_serving_ladder.run --rung quant
# Anywhere with the results:
PYTHONPATH=src:. python -m experiments.gleipnir4b_serving_ladder.compare
```

## Result

Not run yet.
