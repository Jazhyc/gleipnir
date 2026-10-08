# B200 selected reference concurrency scaling

The current FP8/0.31 NC2 measurement is recorded in
[the host comparison below](#current-fp8-scorer-on-nc2).

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

## Current FP8 scorer on NC2

2026-10-08. The same full320 cohort and timing protocol, now using the selected
FP8/vLLM 0.31 recipe on user-provisioned US-NC-2 Pod `mnmqm5d3eiyvuz`.
`current_config.json` freezes c1/2/4/8/16/32/64/128, one excluded full warmup and
three timed passes at each level. Keep 32K context/chunk, 128 engine sequences,
stock FCFS, merged adapter, native frontend and all other precision choices
fixed. Host: Xeon Platinum 8568Y+, 20.4 CPU quota, 250,999,996,416-byte memory
limit, B200 183,359 MiB, driver 595.91.07. No OOD or final-ID evaluation,
threshold fitting, precision selection or promotion.

Fresh-container startup required restoring the complete native tokenizer package
and persistent path registration, six checksum-bound diagnostic generators, and
TVM FFI in the retained CUTLASS overlay. That overlay's 0.1.14.post1 shadowed the
locked 0.1.11 and aborted TileLang import after graph capture. A CPU-only import
reproduces the failure; restoring 0.1.11 fixes it, with original files preserved.
All six overlay package metadata hashes then match the accepted EU receipt.
Scheduler preflight passes 3/3 without skips; native error, layout and QK checks
match the earlier receipt, preserving the known strict MXFP8/BF16 failure.
No runtime kernel source is replaced. The three failed starts are retained as
`nc2_fp8_01/02/03`; source-restoration and dependency-reconciliation receipts
record the actual changes. Staging now guards the dependency override and
automatically restores/registers the tokenizer.

Cold Torch compilation takes 53.03 seconds; the successful retry reuses it in
2.77 seconds and reaches ready in 97.19 seconds. Its adapter canary matches the
accepted FP8 scores exactly (MAE 0, correlation 1), with all 16 native FP8
projections dispatched. Compiler/kernel caches remain persistent.

Medians of the three timed full320 passes, excluding all eight warmups:

| Concurrency | Input tokens/s | Repeat range | p50 | p95 | p99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26,675 | 26,473–26,870 | 149.60 ms | 251.59 ms | 286.62 ms |
| 2 | 46,473 | 46,454–46,930 | 157.69 ms | 281.85 ms | 311.32 ms |
| 4 | 78,918 | 75,933–80,426 | 199.35 ms | 305.99 ms | 427.35 ms |
| 8 | 114,389 | 111,741–115,277 | 273.88 ms | 437.30 ms | 512.64 ms |
| 16 | 157,089 | 151,710–159,531 | 391.53 ms | 641.94 ms | 703.04 ms |
| 32 | 179,248 | 174,354–180,357 | 687.06 ms | 955.08 ms | 1,117.41 ms |
| 64 | 176,492 | 174,336–179,930 | 1,351.07 ms | 1,730.56 ms | 1,856.51 ms |
| 128 | 175,093 | 166,769–176,741 | 2,567.11 ms | 3,077.45 ms | 3,362.89 ms |

Concurrency 32 is the first point within 95% of measured peak. Raising it to
128 decreases the median throughput by 2.32% while increasing p95 3.22-fold;
the higher-level repeat ranges overlap, so do not interpret a precise best
concurrency from these medians. The c1-to-c128 throughput ratio is 6.56 versus
2.26 on the old curve. That full EU curve used 0.24/FP4 attention projections,
so its apparent earlier c16 plateau is confounded by recipe as well as host.
There is no archived full320 lower-concurrency FP8 timing curve.

The matched-recipe control is all six archived FP8/0.31 c128/full320 passes:
210,281 tokens/s median, range 200,626–211,291. NC2 c128 is 16.73% slower;
p95 is 3.077 versus 2.751 seconds, 11.86% higher. This measures an environment
difference, without isolating CPU, driver, storage or GPU as its cause. A runtime
snapshot shows maximum SM clock (1,965 MHz) and no ongoing CPU-quota throttling;
it is insufficient to attribute the low-concurrency slowdown. No resource,
clock, scheduler or kernel settings are changed in response to the results.

Quality deltas against repeat-median scores from all six FP8 c128 controls:

| Concurrency | Pooled AUROC delta | Macro AUROC delta | Pooled pAUROC@20 delta | Macro pAUROC@20 delta | Score MAE | Threshold flips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.28520 pp | +0.24842 pp | +0.04493 pp | -0.29988 pp | 0.04429 | 18 |
| 2 | +0.28520 pp | +0.24842 pp | +0.04493 pp | -0.29988 pp | 0.04429 | 18 |
| 4 | +1.11150 pp | +2.74746 pp | +2.45898 pp | +1.93923 pp | 0.03075 | 12 |
| 8 | +0.31450 pp | +1.18431 pp | +1.28018 pp | +0.55948 pp | 0.01556 | 7 |
| 16 | +0.05274 pp | +0.02776 pp | -0.13674 pp | 0.00000 pp | 0.00236 | 1 |
| 32 | -0.06837 pp | +0.00766 pp | -0.22464 pp | -0.11494 pp | 0.00187 | 1 |
| 64 | -0.03712 pp | 0.00000 pp | -0.24418 pp | 0.00000 pp | 0.00147 | 1 |
| 128 | +0.01563 pp | +0.05075 pp | +0.02930 pp | 0.00000 pp | 0.00185 | 1 |

The c128 source-macro pAUROC@20 is unchanged at 0.779613; pooled pAUROC@20 is
0.614100 versus 0.613807. Lower-concurrency score shifts remain substantial
despite stable or improved ranking. This is the previously observed batch
dependence of the quantized recipe, not strict score parity. These training-seen
diagnostics do not select a quality-improving concurrency. Every repeat, source
breakdown, calibration, ties, undefined single-label source and length-bin
latency remains in the raw receipts.

`results/b200_score_scaling/nc2_fp8_04/` contains all 24 timed passes, eight
warmups, comparisons, audits and log snapshots. All 97 collected run/startup/
validation artifacts pass SHA-256 checks. `current_summary` exports source-bound
`scaling.json`, CSV and standalone PNG/SVG curves, distinguishing the older full
curve from the same-recipe c128 control. Seventeen focused tests pass; Ruff is
clean. API 9865 and sole GPU engine 9927 remain ready and warm, at 169,886 MiB
GPU memory and 0% utilization at closure. Capacity remains running.
