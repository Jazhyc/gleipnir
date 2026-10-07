# B200 serving CPU placement

2026-10-07. GPU-local CPU affinity does not provide a reliable serving gain on
the retained native Gigatoken/Direct FP4 worker. Restore original affinity and
keep the warm API/engine. This tests CPU execution placement; existing host
memory allocations and their NUMA policy are unchanged.

## Protocol

Hypothesis: placing host execution near the GPU, then preventing migration of
the critical submission thread, reduces launch delays. The B200 at PCI
0000:3a:00.0 is on NUMA node 0 (CPUs 0-47,96-143). The container has a 20.4-CPU
quota and permits CPUs 0-191. CPU masks do not reserve cores against other tenants.

Keep API 107583 and engine 107606, native encoding, HTTP requests, selected GPU
arithmetic, model, compiled graph and caches throughout. No restart or new
capacity. Reuse native encoding/startup receipts rather than repeating unchanged
GPU probes. The client keeps its original affinity. Change all live threads
between drained passes, verifying process birth identity and actual masks before
and after each pass; restore saved masks on failure.

`cpu_placement01` compares original masks with both processes on node 0, then
separate groups (API CPUs 0-3,96-99; engine 4-47,100-143). Use three
position-balanced c1/quick64 triplets and all six condition permutations at
c128/full320, after one excluded warmup per mode/workload.

After those candidates fail selection, `cpu_placement02 --hot-thread` compares
original masks with API CPUs 0-3,96-99, the engine leader on CPU 4 alone and its
background threads on 5-47,101-143. Exclude the leader's SMT sibling CPU 100.
Use three alternating c1 and six balanced c128 pairs, with excluded warmups.
Do not reinterpret this as dedicated-core allocation or memory migration.

Freeze timing selection before each run: >0.5% median paired throughput gain,
at least five positive pairs of six, and <=2% c1 median/p95 regression; or >3%
p95 reduction with no median-latency regression and <=0.5% throughput loss.
Require exact c1 scores and absolute macro/pooled development AUROC changes
<=0.1 percentage points. Quality is a guard, not the candidate-ranking objective.
Choose the highest-throughput eligible candidate; otherwise restore original.
The 320 rows are training-seen systems development examples, not final ID.

## Results

All 45 timed passes complete. Throughput changes below are medians of paired
candidate/original ratios; absolute rates are medians of each mode's passes.
The original controls are contemporaneous within each run.

| Placement | Original / candidate input tokens/s | Paired throughput change | Positive pairs | c1 median original / candidate | c1 p95 original / candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both processes on node 0 | 197396 / 196249 | -1.38% | 2/6 | 158.71 / 155.86 ms | 196.66 / 191.28 ms |
| Separate node-0 CPU groups | 197396 / 197070 | -0.26% | 3/6 | 158.71 / 159.49 ms | 196.66 / 194.57 ms |
| Engine submission thread on CPU 4 | 197506 / 196087 | -0.49% | 1/6 | 160.48 / 157.15 ms | 195.33 / 195.35 ms |

All six-pair bootstrap intervals include no change. Small median-latency
reductions do not satisfy the frozen combined selection rule. None is adopted;
these results do not establish that CPU-local placement is intrinsically slower
on other hosts or after different memory allocation at startup.

There is **zero cgroup CPU-quota throttling** during all 30 timed c128 passes.
Engine main-thread runqueue wait is at most 0.722 ms per pass in the first suite
and 1.208 ms in the fixed-core suite. These unprofiled counters do not sample
stacks or GPU waits, but they give no evidence of substantial CPU starvation.
They cannot explain the hundreds of milliseconds of host-before-launch time
observed in the earlier instrumented engine trace. Investigate custom wrapper
preparation, allocation, synchronization and profiler effects next; adding CPU
cores or applying these affinity masks is not a demonstrated solution.

c1 scores/margins are exact in all candidates, with no flips. At c128,
source-macro/pooled AUROC changes in percentage points are +0.0465/+0.0254 local,
+0.0426/-0.0156 split, and 0.0000/+0.0410 fixed-core. Only the local candidate
has a threshold flip, on an already unstable control example. Scheduling-dependent
FP4 score variation is retained; no GPU precision or final-ID promotion follows.

## Artifacts and retained state

`results/b200_inference_benchmark/cpu_placement01` and `cpu_placement02` contain
executed clients/affinity helpers, all predictions, masks and before/after
process/context-switch/runqueue/cgroup telemetry. `cpu_placement_collection01`
collects logs, driver receipts, final server/campaign/precision/compile receipts
and checksums. All **71 collected files** verify locally. Twenty-four focused
tests pass before the first suite; five targeted tests pass after adding leader
affinity. Ruff passes. Tests cover CPU-list parsing, per-thread restoration,
new threads, recycled PIDs, separate leader masks and selection guards.

API/engine remain 107583/107606, healthy on port 8010 with native Gigatoken and
original CPU masks restored. Only the engine owns GPU memory (169068 MiB).
The selected GPU baseline checksum and compile identity are unchanged. Runtime
metadata records the CPU comparison and retained placement separately from GPU
configuration. Do not enable these rejected masks by default.
