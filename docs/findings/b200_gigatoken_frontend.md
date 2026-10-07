# Native Gigatoken serving frontend

2026-10-07. Native Gigatoken is wired into the API renderer and validated on the
selected Qwen3.5-4B stack. Ordinary text requests improve interactive median/p95
latency from **156.35/285.81 ms to 145.13/187.43 ms** against the preceding HF
text controls. Full c128 throughput is **194611 input tokens/s**, 2.56% below
those controls. Keep this frontend optional for latency work; the selected GPU
kernel/precision reference is unchanged.

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

Retain native API/engine **106836/106874**, port 8010, on the existing NC2 B200.
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
causal cost of native tokenization. A same-worker HF/native frontend A/B would
be needed to settle that attribution; no such runtime-switch control is installed.

Artifacts: `results/b200_attention_gdn_serving/gigatoken_native_profile01`,
including trace, exact batch/kernel comparison, gap/first-batch diagnostics,
executed profile/timing clients, predictions and checksum-verified receipts.
Profiling is stopped and the sole native frontend server remains healthy.
