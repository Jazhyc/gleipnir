# Cache-free complete-prompt monitoring on B200

Date: 2026-10-07. **Keep the selected Gigatoken/direct-FROST reference.** The
cache-free prototype removes persistent cache storage. The initial NC2 screen
stalls on its second c128 pass; a subsequent EU-RO-1 screen completes 20 ordinary
full-batch passes without a stall. The old failure remains unexplained, and the
new latency measurements use a different host/driver. This remains an opt-in
experimental path rather than a promoted batch-serving replacement. A subsequent
same-host screen
measures **30.36 ms cached versus 25.71 ms cache-free**; most of the historical
fivefold latency contrast is associated with the host/runtime environment.
See the [matched policy/restart finding](b200_cache_policy_same_host.md) for
throughput, quality, restart and memory results.

## Scope and evidence

For a complete prompt followed by one decision token, continuation storage is
unnecessary. `prompt_only.json` disables prefix caching and chunked prefill,
requires fresh whole prompts within the existing 32,768-token step budget, and
rejects multiple output tokens/completions. All 32 layers return no cache spec.
A runner-only `EncoderOnlyAttentionSpec` supplies zero-storage metadata; cuDNN
attention remains explicitly **causal**, with the selected MXFP8 arithmetic.
Fresh THD K/V replaces cache scatter/gather. GDN uses zero initial state and
omits final state; convolution retains disposable per-call scratch. Within-prompt
K/V and recurrence computation remain necessary. FP4 projections/direct SwiGLU,
BF16 GDN, FP32 gates, native Gigatoken and direct FROST bindings remain selected.

`prompt_only_canary04` passes seven cases: three packed/paged attention cases
(including 32,768 tokens), two zero-state GDN cases, and two convolution cases
using poisoned initial storage. Candidate outputs match the valid original
kernel paths bitwise. A separate sequence-wise FP32 convolution reference has
relative-L2 differences 0.2712%/0.2729%, below its 1% limit. These checks do not
erase the inherited MXFP8 strict FP32 precision failure or its acceptance record.

In `prompt_only03`, runtime RPC reports **zero runner cache bytes, zero layer
cache bytes and no cache specs**; all 32 layers execute. Startup allocated/reserved
memory is 4.48/5.15 GiB. Observed GPU process memory is about 11.65 GiB during
the successful c1 workload and 17.43 GiB at the later stall, versus the selected
server's approximately 165.15 GiB. The large saving mostly removes its preallocated
cache pool; these observations are not a continuous peak-memory measurement.

Two warmed quick64/c1 passes yield median/p95 **133.42/175.20 ms**, versus
149.69/184.19 ms in the three recorded reference passes: median latency falls
**10.87%**. All 64 c1 scores/margins match the reference exactly. The fresh
twenty-row HTTP canary also matches the selected reference exactly and passes
the existing master/dynamic-adapter score gates. Readiness takes 241.42 s;
startup, canary and warmup are excluded from timings.

One full320/c128 pass completes with all 1,310,581 input tokens in 19.59 s,
**66,906 input tokens/s**. This first-use-inclusive number is not a stable warmed
throughput comparison to the reference's 197,530 input tokens/s. The second
pass stops advancing with eight running and 101 waiting requests, 100% GPU
utilization and no new completions. Preserve that partial repeat; do not count
it as completed or infer an intrinsic threefold slowdown. Whole-prompt admission
also changes batching relative to the reference's chunked prefill.

The completed pass permits a bounded development-quality comparison against
all six reference repeats. Source-macro AUROC changes **+0.00782 percentage
points**, pooled AUROC **−0.03125 points**, with no threshold flips. Mean/max
absolute score differences are 0.001017/0.062659. Full source, calibration,
partial-AUROC and tie diagnostics remain in `outcome.json`. This uses the frozen
training-seen systems cohort, not final ID or production traffic.

## Failures and disposition

Preserve `prompt_only_canary01`: its synthetic GDN fixture omitted the real
Q/K normalization. Canary02 corrects this but does not validate convolution.
`prompt_only01` then fails because the pinned conv launcher requires an index
pointer despite the optional-looking API. `prompt_only02` supplies indices but
uses reserved null slot zero, skipping its first sequence and producing nonfinite
scores. Annotate both runs as invalid performance/quality evidence.

Canary03's convolution comparison shares that sentinel mistake and is invalid
as complete admission. Canary04 uses nonzero slots for the original reference,
disables null-slot semantics for the scratch path, and adds an independent
convolution calculation so matching skipped output cannot pass unnoticed.

The final repeated-batch stall remains unresolved. Both installed cuda-gdb and
py-spy are denied by the container's ptrace policy; no specific active kernel was
identified. Future investigation should instrument operator progress within the
worker and reproduce variable whole-prompt batch shapes. Do not attribute the
stall specifically to attention, GDN or FROST without that evidence.

The stalled driver/server is stopped. `startup --name prompt_only_restore01`
restores the selected recipe in **107.08 s**, retaining shared caches, original
reference controls and GPU compile identity
`5a00ce5a0c8398b6b28bd93baa522878dee420dc7c60682c693ae9893bd7fd5d`.
API **112588** / engine **112611** remain healthy and warm on the existing NC2
B200, port 8010, with native Gigatoken and direct FROST bindings. The restored
score canary passes. No control timing rerun, new capacity or final-ID run is
performed. The cache-free implementation is opt-in; `baseline.json` stays
unchanged. Forty-seven focused tests and Ruff pass; feature commit `f9eb405`.

All **556 artifacts** verify locally under
`results/b200_attention_gdn_serving/prompt_only_evidence01/`, including failures,
native checks, completed scores/timings, stall diagnostics, executed sources,
and restoration receipts. The evidence archive SHA256 is
`c1f0c5a209615eed0319b5b3ff7b24852a60bda6bf058256104d88b63c3daf0f`.

## User-requested shutdown, 2026-10-07

The user subsequently requests terminating the B200 and continuing later.
Stop API/engine 112588/112611 and collect six final shutdown receipts. Reverify
all 556 earlier evidence files and the persistent master adapter checksum.
Runpod deletes pod `i243nsg10usytq` with HTTP 204; a fresh get returns HTTP 404
(`pod not found`). No serving or training worker remains on that pod.

Network volume `ixbh81vf9c` (`gleipnir-b200-workspace`) remains **300 GB** in
**US-NC-2**, confirmed by a fresh volume read. Model sources, master adapter,
shared compiler/kernel caches and results remain there. The optional native
tokenizer package is additionally preserved at
`/workspace/gleipnir/.cache/kernels/gigatoken/0.10.0`. Ephemeral merged weights,
staged runtime copies and in-memory plans are discarded; reconstruct/restage
them when separately authorized capacity is available. The selected reference
and unresolved cache-free batch stall are unchanged.

Final local shutdown evidence is under
`results/b200_attention_gdn_serving/pod_termination_20261007/`. The retired pod
record is historical; do not try to reuse its SSH address or worker PIDs.

## EU-RO-1 resume and stability screen

After separately authorized deployment, pod `qobmmj1weyevg1` uses one B200 in
EU-RO-1, CUDA 13 and driver **580.178.04**, versus **595.91.07** on the retired
NC2 host. No cache-free attention/GDN mathematics changes. Add opt-in cooperative
operator tracing and exact ordered token/batch snapshots so a recurrence of the
stall can be diagnosed without ptrace. Six instrumented full320 passes complete;
all tracing timings are excluded because flushed journal writes are expensive.
Restore every patched operator before ordinary measurements; the warm worker
reports zero runner/layer cache bytes and no cache specifications.

Fresh native checks (`prompt_only_canary_euro02.json`) pass all seven cases,
including bitwise attention/GDN comparisons and independent convolution checks.
The preceding canary01 fails at attention import: isolated CUTLASS 4.8 metadata
was present, but its `libs-core` implementation was missing and the locked
4.5.2 site hook took import priority. Install the missing pinned dependency and
add a source-bound isolated import bootstrap; the NVIDIA source hashes match the
recorded recipe. Preserve the failed receipt. Additional direct-FP4 checks pass
rows 4097, 8193, 16385 and 32767. Their harness's global/full-envelope flags remain
false because this is a four-row extension, not a replacement nine-row receipt.

`prompt_only_euro_warm01` completes **20 ordinary full320/c128 passes** plus one
excluded warmup and three quick64/c1 passes with tracing removed. None stalls.
The HTTP score canary passes; all c1 scores and margins match the archived
reference exactly. Measurements summarize the three c1/six timed c128 repeats:

| Metric | Archived NC2 reference | EU-RO-1 cache-free |
| --- | ---: | ---: |
| c1 median latency | 149.69 ms | 28.54 ms |
| c1 p95 latency | 184.19 ms | 119.29 ms |
| c128 input tokens/s | 197,530 | 185,096 |

Latency is client-observed HTTP round-trip time **within the pod**, including
encoding, queueing, host dispatch, GPU work and response handling. It excludes
external client-to-pod network latency. Both cohorts have the same ordered IDs,
prompt hashes, lengths and total token counts. The quick64 median is dominated
by its 46 prompts below 4096 tokens; the seven longest prompts have approximately
122 ms median latency in the new second repeat. The archived process has a much
higher floor even for short prompts. Source inspection finds no deliberate
per-request delay or changed client timing boundaries; its frontend A/B control
adds locks/counters but does not establish the cause of that floor. **Do not
attribute the apparent 5.24x c1 difference solely to cache removal.** Host,
driver, warm runtime and whole-prompt batching confound this comparison.
The archived 149.69-ms reference itself was measured shortly after a fresh NC2
server start; its client completed timing before enabling profiling. The later
fresh NC2 cache-free server still measured 133.42 ms. Process age or accumulated
profiling therefore does not explain the contrast by itself. Prioritize host
CPU/dispatch, driver synchronization and environment differences as hypotheses,
without assigning a cause until a matched comparison or request trace isolates it.

The six c128 rates span **141,893–208,878 input tokens/s**; preserve this spread,
rather than selecting the fastest pass. The median is 6.29% below the archived
reference. Whole-prompt admission differs from its chunked prefill. Score
mean/max absolute differences are 0.002416/0.111789 with zero threshold flips.
Pooled development AUROC changes **+0.00586 percentage points** (0.885763 to
0.885822), source-macro **+0.88862 points** (0.878144 to 0.887030). A four-example
source contributes a +25-point AUROC change and dominates the macro movement;
this does not establish improved generalization. Full calibration, source,
ranking/tie and repeat diagnostics are preserved in `summary.json`. No final-ID
run or default promotion occurs.

The NC2 stall is **unreproduced on this host**, not causally fixed. Keep that
failure and this completed screen separately. API/engine **13964/14005** are
left warm on loopback port 8010; tracing is off. Historical process notes above
refer to retired capacity. Active-turn checks cannot promise future monitoring
without an agent scheduling tool.

Fresh provisioning takes much longer than a warm restart: dependency download
and workspace installation alone take over 20 minutes, followed by staging and
an avoidable CUTLASS repair; first server readiness takes **520.33 s**. Preserve
the complete staged dependency runtime and native tokenizer as
`.cache/runtime-resume/b200-serving-runtime-euro-20261007.tar.zst`, together with
its binding manifest. The dependency archive SHA256 is
`26d7636171604fe81c469ca7aa901add7d204a54264fc4aaff73686aca4d0bc1`;
its remote zstd integrity check passes. Exact binding metadata, executed clients,
failed receipts and trace/quality evidence are collected under
`prompt_only_euro_collection01`, `prompt_only_euro_trace01` and
`prompt_only_euro_warm01` (422 evidence files with a checksum manifest).
The bundle excludes weights and merged artifacts. Restore
under `/tmp` on a compatible CUDA-13 image, validate the manifest/package hashes,
and restore the corresponding pinned source dependencies before running the
normal launcher. Shared compiler/kernel caches are preserved separately as
`.cache/runtime-resume/b200-serving-caches-euro-20261007.tar.zst` (SHA256
`741952c1b18900d85e404e3996a532d7cf5ff5785d2ae865b51cdece5592e746`).
This is not a promise
of a 90-second first start on different hardware. Focused runtime/trace/prompt
and reference checks pass (36 tests) with Ruff.
