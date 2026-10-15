# Async scheduling for the pooling scorer

The saturated-batch profile ([finding](../../docs/findings/b200_kernel_profile.md#saturated-batches-on-the-adopted-stack-saturation01-2026-10-15))
shows 3–7% GPU idle at the 2K c32 and full320 c128 plateaus. Most of it is
step-start input preparation (about 5 ms per step) plus about 1 ms between
steps. The engine runs `--no-async-scheduling`, so neither overlaps the
previous step's GPU work. vLLM 0.31 disables async scheduling by default for
pooling runners, citing performance rather than correctness.

**Hypothesis.** With async scheduling, the engine queues two batches: it
schedules, prepares and launches step N+1 while step N runs. This recovers
most of the idle time at saturation without changing per-batch arithmetic.

**Intervention.** `launch.async_command` replaces `--no-async-scheduling` with
`--async-scheduling`, and the zero-output-token scheduler with
`Vllm031AsyncPoolingScheduler`: upstream `AsyncScheduler` with the same
correction (`num_sampled_tokens_per_step = 0`; the upstream scheduler hash is
checked). Recipe, workers, limits and caches are unchanged.

**Caveat.** vLLM's `_pool` synchronizes the device on steps where no request
finishes, so mid-chunk steps of long prompts cannot overlap.

**Baselines.** The same-GPU synchronous sweeps from 2026-10-15:
`results/b200_score_scaling/twok_stack01` (fixed 2K) and `euro_stack02`
(full320).

**Run.** `python -m experiments.b200_score_scaling.current --async-scheduling`
with `--workload twok` and `full320`: one warmup and three timed passes at
c1–c128, through normal startup and its canary gate.

**Checks.**
- Within each arm, c1 scores repeat bitwise, and a lone request computes the
  same batch either way. Report any c1 difference from the sync baseline.
- Larger batches may be composed differently. Compare full320 quality drift
  against the archived FP8 c128 repeats, as in the baseline.

**Selection rule (frozen).** Propose adoption only if both workloads gain at
saturation (c32–c128 medians) with non-overlapping repeat ranges, c1–c4 are
not slower beyond repeat ranges, and full320 c128 drift stays within the sync
baseline's batch variation.

**Stop** on a startup or canary failure, a scheduler assertion, nonfinite
scores, or completion. Not adopted without the user's decision.

## Results (`twok_async01`, `euro_async01`; EU-RO-1 B200 `GPU-e5eb9962…`)

Both runs passed startup and the canary (MAE 0.0005, the same as sync).
Medians of three timed passes against the same-GPU sync sweeps:

| Concurrency | 2K sync | 2K async | Change | full320 sync | full320 async | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72,491 | 70,216 | −3.1% | 101,868 | 102,154 | +0.3% |
| 2 | 83,092 | 75,880 | **−8.7%** | 119,660 | 113,840 | **−4.9%** |
| 4 | 157,424 | 150,753 | **−4.2%** | 167,766 | 158,852 | **−5.3%** |
| 8 | 204,584 | 203,116 | −0.7% | 207,330 | 206,342 | −0.5% |
| 16 | 253,532 | 253,651 | 0.0% | 224,562 | 226,109 | +0.7% |
| 32 | 259,641 | 248,744 | −4.2%* | 225,820 | 231,035 | **+2.3%** |
| 64 | 258,903 | 265,572 | **+2.6%** | 224,534 | 230,372 | **+2.6%** |
| 128 | 254,711 | 261,160 | **+2.5%** | 223,755 | 229,147 | **+2.4%** |

\* The 2K c32 async passes were 242,888 / 248,744 / 263,784 tokens/s, two of
them with long p95 tails. The full320 c128 async passes include one at
174,479 (others 229,147 / 230,280). Both sync sweeps also had one slow
2K pass each at c32 and c128 (about 228K). These intermittent stalls are
retained and not diagnosed.

- **Saturation.** c64–c128 (and full320 c32) gain 2.3–2.6% with separate
  repeat ranges. That is at the low end of the 3–7% idle estimate.
- **Low concurrency.** c2–c4 lose 4–9% on both workloads, also with separate
  ranges. This matches vLLM's note that async hurts pooling models. The
  engine fills its two-batch queue before collecting outputs, so a finished
  step's scores are returned only after the next batch has been scheduled
  and launched (about 30 ms of CPU launch at saturation sizes). That delay
  lands directly on request latency when few requests are in flight.
- **Scores.** c1 scores are bitwise equal to sync on both workloads. From c2
  up, batches form differently, so 2–77 scores per pass differ, as expected
  from batch-composition dependence. full320 c128 drift against the archived
  FP8 repeats is MAE 0.0018 with 1 flip, versus 0.0012 with 1 flip for sync.

**Decision.** The frozen rule fails because c2–c4 are slower beyond repeat
ranges. Async scheduling is not adopted; the default keeps
`--no-async-scheduling`. Recovering the saturation gain without the
low-concurrency cost would need an engine change, such as collecting
finished outputs before launching the next batch, or switching modes by
queue depth. That is not attempted here.
