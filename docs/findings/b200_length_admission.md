# B200 length-aware admission

2026-10-07. The initial four-bucket, 250-ms FIFO-promotion policy does not meet
its frozen screening rule. Keep upstream FCFS selected. The modest low-load
short-request benefit is accompanied by longer long-request tails; overloaded
traffic does not improve. This does not establish that FCFS is optimal.

## Contract and runtime

Use the repaired selected FP4/MXFP8/Gigatoken/direct-host scorer on the existing
EU-RO-1 B200, frozen quick64/full320 systems cohorts and every saved selected
repeat. The candidate changes waiting/skipped queue admission only, retaining
running-request scheduling, token budgets, caches, classifier and precision.
See [the experiment](../../experiments/b200_length_admission/README.md) for the
predeclared buckets, age limit and screening thresholds. Full320 has 320 prompts
and 1,310,581 tokens; it is training-seen development, not held-out quality.

Run three quick64/c1 and six full320/c128 warm candidate repeats. Separately,
freeze full320 order and seed-17 Poisson arrivals at 40 and 80 requests/s, with
three repeats after excluded warmups for each policy. This new FCFS control is
needed because archived closed-loop controls cannot measure arrival fairness.
All twelve arrival passes satisfy the 5-ms p95 dispatch-lag gate; the maximum
is 1.805 ms. Latency includes arrival-to-dispatch lag and localhost HTTP serving.

The previous training resident worker held GPU memory despite being idle. After
explicit user approval, verify its identity, retire PID 22966 and preserve its
training artifacts. Package reorganization invalidated source-bound inference
receipts: restore the recorded CLI/test/lint project metadata and exact serving
sources, retaining the current files as backups. Rehydrate the native helper
dependency closure by checksum, without restaging or changing dependencies.
The import bootstrap routes legacy names to those frozen sources and retains
pinned CUTLASS selection. Failed attempts `fcfs_start01`--`06` are preserved:
metadata/source guards, then a scheduler override missing vLLM's throttle
argument. Fix that signature and add the real engine-call regression test.
Handle exited API zombies without treating them as live stop targets.

Keep CPU policy/source hashes outside `additional_config`'s GPU compilation
key. `fcfs_start07` becomes ready in 120.61 s; FCFS and candidate reuse the exact
selected `f0290e9cc3` graph/AOT identity. All precision/classifier audits pass.
Both adapter canaries remain finite with nonzero effect and accepted-reference
agreement; inherited strict master/native failures remain explicit. No reference
throughput timing controls are rerun for this candidate.

## Measurements

Warm medians, using all available archived repeats:

| Metric | Selected reference | Length-aware | Change |
| --- | ---: | ---: | ---: |
| c1 quick64 input tokens/s | 97,272 | 98,748 | +1.52% |
| c1 request p50 | 31.225 ms | 30.480 ms | -2.39% |
| c1 request p95 | 128.537 ms | 126.965 ms | -1.22% |
| c128 full320 input tokens/s | 215,707 | 216,945 | +0.57% |
| c128 request p50 | 2260.057 ms | 2260.497 ms | +0.02% |
| c128 request p95 | 2615.344 ms | 2872.826 ms | +9.85% |

Mixed-arrival medians of three repeat percentiles:

| Offered requests/s | Length bin | FCFS p95 | Length-aware p95 | Change |
| --- | --- | ---: | ---: | ---: |
| 40 | <4K | 262.561 ms | 243.408 ms | -7.29% |
| 40 | 4K--16K | 266.803 ms | 255.083 ms | -4.39% |
| 40 | >=16K | 305.453 ms | 336.208 ms | +10.07% |
| 80 | <4K | 1923.334 ms | 1938.404 ms | +0.78% |
| 80 | 4K--16K | 1967.467 ms | 1989.025 ms | +1.10% |
| 80 | >=16K | 1913.465 ms | 1929.470 ms | +0.84% |

At offered 40 requests/s, both policies complete 38.009 requests/s and about
155,669 input tokens/s including tail drain. At 80, FCFS reaches 52.870 requests/s
and 216,533 input tokens/s; length-aware reaches 52.713 and 215,889. Low-load
achieved throughput is arrival-limited, not GPU peak throughput.

At c1, every paired score/margin and defined AUROC is unchanged. At c128, pooled
AUROC shifts -0.13869 percentage points and source-macro +0.00105 pp. Nonzero
per-source shifts are harm-pressure knowledge +0.61728 pp, insider trading
-0.88889 pp, soft trigger +0.41551 pp and bash arena -0.11338 pp; other defined
sources are unchanged. Undefined single-label sources remain explicit.
Repeat-median score mean/max drift is 0.004715/0.076474, margin mean/max drift
0.048438/0.625. One threshold flip occurs at soft-trigger row 1464; two unstable
baseline-threshold rows are separately retained. Batch composition can change
the accepted quantized stack's rounding even without changing its arithmetic
recipe; these sequential controls do not isolate policy as the sole cause.
The subsequent [stock reference scaling sweep](b200_score_scaling.md) also
finds c128 pooled/macro shifts of -0.23246/-0.10196 pp and one flip against the
same archived control. That reinforces the attribution limit: numerical drift
is observable with stock FCFS too.

The frozen screen fails short p95 improvement (requires >=10%), long p95 bound
(<=10%) and pooled AUROC bound (<=0.1 pp). Throughput, c1 latency and arrival
timing gates pass. Keep the implementation experimental; no automatic promotion
or tuning on final ID. This test supports a small low-load prioritization effect,
not a general scheduler speedup or a production latency guarantee.

## Artifacts and closure

`results/b200_length_admission/length01/` contains the full throughput/quality
receipts and `screen.json`; `fcfs_arrivals01/` and `length_arrivals01/` contain
arrival contracts, all scores/times and per-rate comparisons. Preserve startup
failures, frozen source snapshots/import bootstrap, backups, runtime metadata
reconciliation and training retirement. All 251 collected artifacts match their
remote checksums. Thirty focused CPU tests pass with Ruff clean; local NVML and
TorchScript deprecation warnings are recorded in the check log.

Retire API 27022/engine 27045, preserve caches and restore 19 current source
files and project metadata, removing 96 temporary legacy files. GPU closure is
0 MiB, 0% utilization, no compute processes; do not restart a reference merely
because this candidate was rejected. The user's subsequent explicit request for
a seven-level reference scaling benchmark authorizes a separate reference launch.
