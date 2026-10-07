# B200 selected reference concurrency scaling

2026-10-07. On this fixed workload, stock FCFS reaches 98.88% of measured peak
throughput at client concurrency 16. Raising concurrency from 16 to 128 adds
1.13% throughput while p95 latency grows 5.68-fold. This is a throughput/latency
operating-point measurement, not proof of an optimal scheduler or a production
arrival-rate SLO.

## Contract

The user explicitly requested fresh measurements at concurrency 1, 2, 4, 16,
32, 64 and 128. Use one persistent selected repaired score server on the same
EU-RO-1 B200. Preserve native Gigatoken/direct FROST, FP4 MLP/projections, MXFP8
prefill, BF16 operands/KV and FP32 gates/state, exact two-row BF16 classifier,
causal LAST pooling, chunked prefill, prefix caching off, 32768-token budget and
128 engine-sequence cap. These are client concurrency levels; the engine cap
is not varied. See [the experiment](../../experiments/b200_score_scaling/README.md).

Use the same frozen full320 order, prompts, labels and 1,310,581 input tokens at
every concurrency. Earlier quick64/c1 measurements are a different cohort.
Exclude one full warmup at each level, then run three timed passes with fresh
HTTP pools. Closed-loop request latency starts after acquiring the client
semaphore; it includes localhost HTTP, encoding, engine queueing and inference.
There are no external-network delays or prefix-cache hits. This training-seen
systems cohort is not held-out quality or observed production traffic.

Rehydrate the exact source-bound serving recipe through the checked snapshots
and bootstrap from the [admission trial](b200_length_admission.md); retain current
checkout backups. No dependency/compiler cache reset or numerical gate bypass.
The stock reference becomes ready in 112.50 seconds and reuses `f0290e9cc3`.
Classifier/precision audits and adapter canary pass (mean score error 0.000807,
correlation 0.999967, finite output, nonzero adapter effect). Preserve inherited
strict master/native failures separately from accepted finite quality.

## Measurements

Medians across three timed repeats per level, same workload throughout:

| Client concurrency | Input tokens/s | Requests/s | p50 latency | p95 latency | p99 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96,046 | 23.45 | 30.47 ms | 130.74 ms | 172.67 ms |
| 2 | 110,873 | 27.07 | 56.56 ms | 168.97 ms | 207.00 ms |
| 4 | 157,242 | 38.39 | 85.07 ms | 221.20 ms | 271.32 ms |
| 8* | 196,514 | 47.98 | 153.28 ms | 303.93 ms | 361.99 ms |
| 16 | 214,332 | 52.33 | 288.13 ms | 469.71 ms | 541.00 ms |
| 32 | 216,528 | 52.87 | 598.26 ms | 760.70 ms | 779.03 ms |
| 64 | 216,498 | 52.86 | 1185.45 ms | 1351.00 ms | 1466.38 ms |
| 128 | 216,761 | 52.93 | 2260.55 ms | 2669.56 ms | 2695.00 ms |

*The user subsequently requested c8 during the
[fixed 2K sweep](b200_context_scaling.md). Its full320 addendum uses this same
warm process, inputs and timing protocol; repeat range 195,353–197,887 input
tokens/s. The original `scale01` report and quality comparisons remain unchanged.
The 2K finding includes a figure with all eight points on both workload curves.

Input-throughput repeat ranges are 95,908--96,931; 109,683--111,918;
153,360--157,960; 213,997--216,269; 216,256--216,699; 216,155--218,268;
and 214,555--217,095 tokens/s, respectively. Small differences above concurrency
16 are within the measured repeat spread; do not infer a precise best level
from the 128 median alone. Most of the 2.26-fold c1-to-c128 throughput gain is
already achieved by c16, while queueing increases latency. The source distribution
and token budget are fixed, so this does not compare alternative batch budgets
or engine sequence caps.

## Quality and interpretation

Pair each level's three full320 prediction repeats with all six archived selected
c128/full320 repeats. These are quality/batch-shape comparisons, not matched
lower-concurrency timing controls. Report AUROC changes in percentage points:

| Concurrency | Pooled AUROC delta | Source-macro delta | Repeat-median threshold flips |
| ---: | ---: | ---: | ---: |
| 1 | +0.20706 pp | +0.11405 pp | 7 |
| 2 | +0.20706 pp | +0.11405 pp | 7 |
| 4 | +0.28911 pp | -0.19830 pp | 6 |
| 16 | -0.09962 pp | +0.01782 pp | 1 |
| 32 | -0.21292 pp | +0.36814 pp | 1 |
| 64 | -0.11916 pp | +0.01381 pp | 1 |
| 128 | -0.23246 pp | -0.10196 pp | 1 |

The accepted quantized stack is not batch-invariant. Per-source changes,
undefined single-label sources, calibration, score ties, unstable baseline
thresholds and all raw prediction repeats remain in `c*_comparison.json`.
Fresh stock FCFS at c128 also drifts against its archived control; the admission
trial's numerical shift therefore does not isolate its policy as the sole cause.
No final-ID evaluation, precision promotion, threshold change or recipe change.

## Artifacts and closure

`results/b200_score_scaling/scale01/` retains all 21 timed repeats, seven excluded
warmups, quality comparisons, canary, manifest, source snapshots, reference
selection and cache/native receipts. All 68 collected remote artifacts match
their checksums. `scaling.json`/`scaling.csv` export the medians/ranges; standalone
`scaling.png`/`scaling.svg` plot throughput and latency, with min/max repeat shading.
The tracked `summarize` entrypoint binds its source and raw report by checksum.
Three focused contract/export tests pass and Ruff is clean.

User-requested reference closure: API 27565, sole GPU engine 27588, ready with
172288 MiB allocated and 0% utilization after completion. Keep that reference
warm, compiler/kernel caches intact and frozen serving sources available for
its bound runtime. Current checkout/project-metadata backups remain under the
admission source-reconciliation artifacts for restoration when this service is
retired. No capacity creation, stop or termination occurs.
