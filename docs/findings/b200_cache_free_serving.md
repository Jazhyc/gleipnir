# Cache-free complete-prompt monitoring on B200

Date: 2026-10-07. **Keep the selected Gigatoken/direct-FROST reference.** The
cache-free prototype improves interactive latency and removes persistent cache
storage, but its repeated c128 batch run stalls. It is an experimental path,
not an accepted batch-serving replacement.

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
