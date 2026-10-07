# Native Gigatoken serving frontend

2026-10-07. Native Gigatoken is wired into the API renderer and validated on the
selected Qwen3.5-4B stack. The latest same-worker comparison improves interactive
median/p95 latency from **152.62/270.04 ms to 148.88/191.96 ms**. Full c128
throughput is **196523 versus 198633 input tokens/s**; the median paired change
is -1.11%, with the final two pairs essentially tied. Keep this frontend optional
for latency work at this stage. The user subsequently selects native Gigatoken
plus direct FROST bindings as the
[standard combined reference](../decisions/b200_gigatoken_direct_host_reference.md).
The GPU kernel/precision recipe remains unchanged.

## Encoding, compatibility and the bulk headline

The pinned optional wheel is Gigatoken **0.10.0**, installed into
`/tmp/gleipnir-gigatoken-0.10.0` without changing the locked serving environment.
The project documents Qwen3.5 support and
[native and compatibility APIs](https://github.com/marcelroed/gigatoken).
Its approximately 1000x headline concerns large-file, multicore throughput. The
Qwen3.5 rows report 558x on the 144-core EPYC and 994x on M4 Max; those are not
single-request latency measurements. HF compatibility mode has additional
Python/return-layout work, but that alone does not explain the headline gap.

The first probe preserves exact IDs on all frozen 320 prompts and 99 synthetic
Unicode/special-token/truncation cases. Three warmed serial passes using the
compatibility encoder take a median **0.148 s**, versus **2.809 s** for the
serving cached HF tokenizer (about 19x). A first un-warmed Gigatoken cohort pass
takes 0.147 s; construction separately takes 0.967 s. Warm median encode time
for 16k+ prompts falls from 46.51 to 1.48 ms. Do not fold construction into a
steady-state request or advertise a cold-process startup improvement.

A follow-up isolates native API overhead on the same full320 strings, checking
exact IDs for every output. Five warmed repetitions give:

| API | Full-cohort median seconds |
| --- | ---: |
| HF compatibility, individual inputs | 0.0935 |
| Native individual inputs, NumPy result | 0.0457 |
| Native individual inputs, Python ID lists | 0.0632 |
| Native batch, Python lists, serial | 0.0951 |
| Native batch, Python lists, 20 Rayon threads | 0.0981 |

Native scalar/list encoding is about 1.48x faster than compatibility in this
separate check. Batching/20 threads does not help this small workload. These
CPU timings have their own process/cache state; do not divide the earlier HF
pass by a later native measurement and call that a matched speedup.

## Two live serving paths

`gigatoken01` first encodes each request in the caller, on one thread, and sends
IDs to the unchanged server. Encoding, thread handoff and ID JSON are inside
request timing. c1 median/p95 becomes 149.75/195.12 ms, with exact score parity,
but c128 throughput regresses 4.16% (199731 to 191425 tokens/s). The observed
median encoding-plus-queue wait at c128 is 37--47 ms versus about 0.13 ms inside
the encoder itself. This establishes a queue/handoff cost in that path, not a
causal attribution of its entire throughput regression.

The user then requested direct integration. `gleipnir.serving_gigatoken`
replaces only `BaseRenderer._encode` with native `Tokenizer.encode().tolist()`.
Retain the original HF tokenizer for metadata, decoding and all other methods,
and vLLM's existing single-worker renderer pool. No HF compatibility object or
caller-side tokenization is used in this variant. Admit token-neutral ByteLevel
postprocessing, preserve left/right and zero-length truncation, and reject
unsupported flags. Construction occurs once per renderer before readiness.

The wrapper imports the original GPU registry at module scope for spawned
engines. Bind CPU frontend source, optional package files and validation receipt
separately in `server.json`; the GPU model/config/source identity is unchanged.
Frontend reuse checks include these bindings. Changes to CPU admission must not
silently reuse a different resident frontend. Stop the prior API/engine
105110/105163 before launching the replacement.

`gigatoken_native01` loads the same cached AOT entry and compile identity
`5a00ce5a0c8398b6b28bd93baa522878dee420dc7c60682c693ae9893bd7fd5d`.
Readiness is 94.82 s, and the twenty-example serving score canary passes exactly.
The live receipt confirms native encoding, one renderer worker and left
truncation. The new helper passes full320 and 99 fixture comparisons; the live
`/tokenize` endpoint also matches all 320 ID sequences exactly.

Three warm ordinary-text passes, after one excluded warmup per concurrency:

| Measurement | Previous HF text | Native API frontend | Change |
| --- | ---: | ---: | ---: |
| c1 quick64 median latency | 156.35 ms | 145.13 ms | -7.18% |
| c1 quick64 p95 latency | 285.81 ms | 187.43 ms | -34.42% |
| c1 pass throughput | 25713 tokens/s | 30703 tokens/s | +19.41% |
| c128 full320 throughput | 199731 tokens/s | 194611 tokens/s | -2.56% |
| c128 median latency | 2.402 s | 2.417 s | +0.61% |
| c128 p95 latency | 2.753 s | 2.922 s | +6.13% |

The HF controls are preceding same-host runs, because replacing the frontend
requires stopping the old server. They are not simultaneous or interleaved
controls; host/run variation remains a limitation. Native c128 pass rates are
196665/194611/191993 tokens/s. The CPU encoder is faster, but a robust saturated
throughput improvement is not established. No new GPU profile attributes the
remaining difference to batch formation, launches or HTTP work.

c1 scores/margins are exactly equal, with no AUROC or threshold changes.
c128 changes source-macro/pooled AUROC by **-0.0030/-0.0762 percentage points**;
mean/max score difference is 0.002315/0.123876. Two threshold flips are both on
already unstable HF-control examples. Exact IDs and unchanged arithmetic with
different scheduling do not ensure equal batched FP4 scores. This is the frozen
training-seen systems-development population; final ID is not used or promoted.

## Artifacts and retained state

`results/b200_inference_benchmark/` holds `gigatoken01`, `gigatoken_api01`, both
native canaries, `gigatoken_native_quality01`, its executed diagnosis driver,
and `gigatoken_collection01`. Startup/source archives are under
`results/b200_attention_gdn_serving/gigatoken_native01`. All **141 collected
files** verify locally by checksum. The collected PyPI wheel SHA256 is
`2f3481d0c067beacf1c0a7640d956edbc8973014ba8cfa3e18bfafb02f78e53b`;
19 installed wheel files match, with installer-generated RECORD metadata
recorded separately. Thirty native/launcher/cache tests and eight caller tests
pass; Ruff passes. The restricted local threaded test stalled in event-loop
waiting and was stopped; the caller tests passed on the pod.

At this stage, retain native API/engine **106836/106874**, port 8010, on the existing NC2 B200.
Only the engine owns GPU memory. CPU frontend selection remains opt-in via
`startup --frontend-validation
results/b200_inference_benchmark/gigatoken_native_canary02/validation.json`.
Selected GPU baseline SHA256 remains
`39811c43e0b4bbf574e682d7b21f09e394909af3af4a69f3b398193cace89166`.

## Throughput decline diagnostic

At the user's request, capture one native-frontend full320/c128 engine trace
on the retained worker and compare with the archived unchanged-GPU-recipe
`native_fp4_output_profile01`. This does **not** confirm an intrinsic Gigatoken
regression: the profile control is older, and the benchmark controls preceded
the server restart. A smaller-batch GPU-efficiency explanation is not supported
by this trace.

| Instrumented engine observation | Archived reference | Native frontend |
| --- | ---: | ---: |
| Nonempty physical GPU batches | 42 | 41 |
| Full 32768-token batches | 39 | 39 |
| Other nonempty batches | 1428 / 29116 / 2085 | 1428 / 31201 |
| CUDA kernel count | 44784 | 43688 |
| Summed GPU kernel time | 5.852 s | 5.802 s |
| First-to-last kernel window | 6.425 s | 6.513 s |
| Kernel-free gaps | 574.57 ms | 713.09 ms |
| Copies within those gaps | 88.14 ms | 88.31 ms |
| Before-next-launch time, excluding copies | 445.55 ms | 579.21 ms |

Both schedule exactly 1310581 input tokens. Exclude two zero-token execution
annotations from each physical-batch count. Native does slightly less GPU work,
but has more time before the host submits its next kernel. 98.86% of native
kernel gaps overlap engine execution contexts; the API/tokenizer is untraced.
This is temporal overlap, not causal attribution to a particular CPU operator.

Almost all additional gap time is concentrated in the first 1428-token context:
gap overlap rises from 194.33 to 336.79 ms (+142.47 ms), while total gaps rise
138.52 ms. CPU context duration rises from 217.04 to 352.79 ms. Several large
gaps overlap original FP4 preparation/GEMM or GDN scopes. This points toward
first-use/host-execution or profiling variation; it does not prove a cold plan,
compiler-cache miss or tokenizer-induced scheduling penalty.

The instrumented native HTTP pass takes 6.7494 s versus archived 6.7249 s
(+0.36%). Exclude both from speed claims. A justified short unprofiled follow-up
after this shape has executed gives 195081/190842/196593 input tokens/s, median
195081. That remains 2.33% below the preceding HF text control, but only 0.91%
below the separately recorded five-pass standard GPU reference (196866).
Warming this profiled shape does not establish recovery or the cause of the
difference. Do not present the original 2.56% observation as a demonstrated
causal cost of native tokenization. The same-worker HF/native frontend A/B
follow-up is recorded below.

Artifacts: `results/b200_attention_gdn_serving/gigatoken_native_profile01`,
including trace, exact batch/kernel comparison, gap/first-batch diagnostics,
executed profile/timing clients, predictions and checksum-verified receipts.
Profiling is stopped and the sole native frontend server remains healthy.

## Same-worker HF/native comparison

The user requests the comparison and asks whether additional host time offsets
Gigatoken's benefit. Restart once into `gigatoken_ab_start01`, then retain the
same API **107583** and GPU engine **107606** for all conditions. An opt-in
`--frontend-ab` controller routes encoding to cached HF or native Gigatoken
within the same renderer thread pool. Both paths use the same controller timing
and locking. Switch only after drained HTTP passes, verify callback counts and
both PIDs, and require exact IDs for all 320 prompts in each mode. Original HF
metadata/decode, HTTP text requests, GPU arithmetic, graph and caches stay fixed.

All 18 timed passes complete: three alternating c1/quick64 pairs and six balanced
c128/full320 pairs, after two excluded warmups per mode/workload. There is no
caller-side encoding. Latency and tokens/s below are medians across each mode's
passes; the throughput change uses the median of paired native/HF ratios.

| Measurement | HF | Native Gigatoken | Change |
| --- | ---: | ---: | ---: |
| c1 median latency | 152.62 ms | 148.88 ms | -2.45% |
| c1 p95 latency | 270.04 ms | 191.96 ms | -28.91% |
| c1 input tokens/s | 26065 | 30099 | paired +15.18% |
| c128 input tokens/s | 198633 | 196523 | paired -1.11% |
| c128 median latency | 2.394 s | 2.404 s | +0.38% |
| c128 p95 latency | 2.783 s | 2.831 s | +1.72% |
| Encoder seconds per full320 pass | 5.041 | 0.143 | 35.2× faster |

c128 paired changes are -1.66/-2.07/-1.15/-1.08/+0.05/-0.07%. The final two
pairs are essentially tied. A seeded bootstrap of the six paired ratios gives
a median-change interval of -1.865% to -0.010%; six repetitions on this frozen
workload do not establish a universal production penalty. The same-worker
comparison reduces the earlier apparent 2.56% decline and confirms the latency
benefit, without demonstrating saturated throughput improvement.

Encoder timing sums callback wall durations, not process CPU utilization or
additional end-to-end elapsed time. Encoding overlaps engine execution at c128;
saving about 4.90 encoder-seconds therefore does not imply a 4.90-second serving
gain. At c1, latency directly benefits, particularly on longer prompts.

### Matched engine profiles

After timing, profile HF/native/HF on the same worker. Exclude all instrumented
HTTP times from speed results. Preserve the first HF activation separately
(42 physical batches, 44784 kernels); compare native against the later warm HF
activation below. Each schedules exactly 1310581 input tokens and 39 full
32768-token batches, excluding two zero-token execution annotations.

| Engine observation | Warm HF | Native |
| --- | ---: | ---: |
| Nonempty physical GPU batches | 41 | 41 |
| Other nonempty batch tokens | 398 / 32231 | 806 / 31823 |
| CUDA kernels | 43688 | 43688 |
| Summed kernel time | 5.80178 s | 5.77727 s |
| First-to-last kernel window | 6.40544 s | 6.46341 s |
| Kernel-free gaps | 605.09 ms | 687.57 ms |
| Copies within gaps | 88.58 ms | 88.71 ms |
| Before-next-launch time, excluding copies | 474.19 ms | 553.46 ms |
| First nonempty context gap overlap | 256.52 ms | 280.83 ms |

Native has 24.51 ms less summed kernel work but 79.27 ms more time before the
next host launch, excluding copies. Its GPU window grows 57.97 ms. Thus additional
host-submission delay does offset the small GPU-work saving in this matched
trace. Unlike the old diagnostic, the first small context explains only 24.31 ms
of the 82.48 ms additional total gaps. Shapes still differ despite equal batch
and kernel counts. The engine-only profiler omits API tokenization, so these
intervals do not identify the responsible host operator or prove that the native
encoder intrinsically adds engine overhead. One native trace is insufficient
for that causal claim. Instrumented HTTP times are 6.8383 s native versus 6.8086 s
warm HF (+0.44%), rather than the unprofiled paired estimate.

### Quality, artifacts and retained worker

c1 scores and margins remain exactly equal, with zero flips and AUROC deltas.
c128 source-macro/pooled AUROC changes by **-0.0348/-0.0547 percentage points**.
Mean/max score difference is 0.001520/0.062298; margin difference is
0.017188/0.437500. One threshold flip is on an already HF-unstable example.
Exact IDs plus unchanged FP4 arithmetic can still produce scheduling-dependent
scores. This is the frozen training-seen development population; no final-ID
selection or precision promotion occurs.

Artifacts: `results/b200_inference_benchmark/gigatoken_ab01` includes the executed
client, all predictions, traces and executed profile-analysis script;
`gigatoken_native_canary03` binds exact native/HF token parity and 99 fixtures;
`results/b200_attention_gdn_serving/gigatoken_ab_start01` records startup and
source snapshots. Readiness is 93.31 s and the twenty-row score canary is exact.
All **122 collected files** in `gigatoken_ab_collection01/checksums.json` verify
locally. Thirty-four focused tests and Ruff pass. Source/package hashes are
reverified at closure; the selected GPU baseline checksum and compile identity
remain unchanged.

All profiles are stopped, native mode is restored (generation 70), and the sole
GPU engine remains healthy and warm on port 8010. Campaign state records API/engine
107583/107606, native frontend, validation and artifact paths. The A/B switch is
opt-in and disabled for ordinary launches; use the guarded `ModeController`
rather than sending signals to an arbitrary server. Gigatoken remains useful for
interactive latency, with no established saturated-throughput gain.

## Host optimization diagnosis

The user's follow-up asks whether the host itself gets slower and what to
optimize. Distinguish delayed engine submissions from slower CPU computation.
The same native encoder is faster; the trace reports kernel-free intervals
before CUDA calls, which can include Python/native wrapper work, waits,
descheduling and profiler effects. It does not establish CPU saturation or
an intrinsic host slowdown caused by Gigatoken.

Reanalyze the existing warm HF and native traces without a new GPU run. Classify
the 474.19/553.46 ms before-launch intervals excluding GPU copies using the
innermost recorded CPU/API scope on the launching engine thread. Intersections
are exclusive and sum back to the original interval totals. They are temporal
overlaps, not sampled CPU stacks or causal assignments:

| Scope overlapping the gap | Warm HF | Native |
| --- | ---: | ---: |
| Untraced host work/wait | 154.42 ms | 168.27 ms |
| `gleipnir::frost_inference_linear` | 74.81 ms | 117.56 ms |
| `vllm::qwen_gdn_attention_core` | 28.17 ms | 39.22 ms |
| `gleipnir::norm_frost_pack` | 9.20 ms | 26.61 ms |
| `aten::empty` | 12.71 ms | 15.36 ms |

The FROST wrapper scope contributes 42.75 ms of the 79.27 ms difference in
overlapping time. Thus investigate the engine's custom operator wrappers and
their input/descriptor preparation rather than treating faster encoding as
the identified source. This scope can include unrecorded nested native work
and profiling overhead. Optimized GPU kernels do not remove host argument
packing, tensor views, allocation and submission costs.

Read-only CPU inspection finds a 20.4-CPU cgroup quota with affinity to all 192
logical CPUs. The B200 is local to NUMA node 0 (CPUs 0-47,96-143); API and engine
were observed on CPUs 146/174 on node 1. This is one idle snapshot, not proof of
their placement during timed passes. No CPU pressure is present at collection;
historical throttling counters do not diagnose the timed workload. No affinity
or memory policy is changed. The installed vLLM has NUMA-binding support;
[upstream guidance](https://docs.vllm.ai/en/stable/configuration/optimization/#numa-binding-for-multi-socket-gpu-nodes)
describes binding GPU worker execution and memory at process startup.

Prioritize a matched GPU-local CPU-affinity trial, then reduce measured wrapper
overhead by reusing stable descriptors and bounded workspaces where lifetimes
and concurrent execution permit it. Current native SwiGLU allocates packed
outputs plus unused descriptor storage each call; MXFP8 allocates packing and
scratch workspace per call. Do not share mutable outputs across in-flight
requests or graph replays. Verify effective async scheduling before proposing
it as an additional optimization; available configuration alone does not prove
it is disabled. A CPU-stack/OS-runtime trace would distinguish untraced work
from waits; nsys, py-spy and perf are absent on the current pod.

The earlier large-prefill piecewise graph trial already reduced launch API
counts but regressed throughput and latency. Do not repeat it unchanged or
assume fewer calls will help. In this native profile, eliminating all 553 ms
of before-launch time would yield an optimistic 9.4% GPU-window speedup with
kernel work fixed; this is an upper bound, not a serving forecast.

Artifacts: `results/b200_inference_benchmark/gigatoken_host01` contains the
executed offline analysis, exact input trace hashes, exclusive scope totals and
CPU inventory. The retained server stays healthy in native mode; this diagnosis
does not change code, kernels, affinity or the selected reference.

The subsequent authorized [CPU-placement comparison](b200_serving_cpu_placement.md)
tests socket-wide local affinity, disjoint API/engine groups and a fixed engine
submission core across 45 timed passes. None meets the frozen combined timing
rule; original masks are restored. There is no quota throttling during timed
c128 passes, and engine main-thread runqueue waits remain below 1.21 ms per pass.
This gives no evidence of significant CPU starvation and redirects investigation
toward wrapper work/waits rather than adopting a CPU-local mask.
