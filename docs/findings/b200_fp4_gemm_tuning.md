# Shape-dependent FP4 GEMM tiles on B200

Date: 2026-10-06. A bounded NVIDIA tile/cluster sweep finds small native gains
for output projections, but no clear whole-serving win. The existing tile wins
every tested large-row shape. After reviewing the tradeoff, the user explicitly
selects this stack as the optimization reference. See the
[selection decision](../decisions/b200_tuned_fp4_inference_baseline.md).

| Metric | Previous combined preparation | Tuned candidate |
| --- | ---: | ---: |
| Five-pass warm c128 input tokens/s | 185,767 | 182,682 (−1.66%) |
| Saved/warm c1 median latency, ms | 149.47 | 147.00 |
| Saved/warm c1 p95 latency, ms | 286.38 | 270.26 |
| Source-macro AUROC, saved full-sweep vs candidate warm repeats | 0.885192 | 0.884608 (−0.0584 pp) |
| Pooled AUROC, same score comparison | 0.908013 | 0.908150 (+0.0137 pp) |

Latency controls come from saved c1 passes; the candidate receives a further
c1 confirmation after the complete sweep and warm c128 workload. Its initial
post-startup median was 163.41 ms. Warmup history and ordinary serving variation
are not randomized here, so the small latency difference is a hint rather than
proof of a tile-induced speedup. Native GPU timing does not measure HTTP,
dispatch, batching or whole-model latency.

## Frozen intervention and selection

Use the existing NC2 B200, merged BF16 model, combined vendor/normalization/
SwiGLU FP4 preparation, FP4 MLP and large GDN projections, cuDNN MXFP8 attention
prefill, BF16 small linears/cache/recurrence/decode and FP32 gates/state. Retire
the previous combined API/engine 88949/89008 before native work. No new capacity
is launched or terminated.

Change NVIDIA FROST tile geometry on the existing projection graph, preserving
the raw BF16 matmul rounding and row-descaling epilogue. Use six fixed SM100
candidate configurations, eight row counts (1/17/129/1536/2304/4096/29184/32768)
and four fixed K/N shapes. One resident native worker reuses packed weights,
plans and shared disk caches. Select separately at M<=4096 and M>4096 only if
all eight numerical/isolation/replay checks pass and geometric-mean GPU gain
reaches 2%. Native timing includes the row-scale helper and GEMM but excludes
activation packing. Each timing is the median of five samples of 32 CUDA-graph
replays; exclude graph warmup. Small-band selection uses M129/1536/2304/4096;
the remaining tiny rows are admission checks and recorded timing diagnostics.

`fp4_gemm_tune01.json` completes in 58.4 seconds. All 160 executable cases
have zero relative-L2 difference versus the original tile, exact zero rows,
unchanged other rows and matched changed-input graph replay. The M256/N256
configuration fails upstream compilation for all four shapes: per-GEMM TMEM
budget 496 columns is less than the 512-column accumulator requirement.
Preserve these four compile rejections. No precision tolerance is widened.

| Projection | K × N | Selected M<=4096 tile M × N | Native small-band speed ratio | M>4096 |
| --- | --- | --- | ---: | --- |
| MLP gate/up | 2560 × 18432 | 128 × 256 | 1.000 | Original tile |
| GDN input | 2560 × 12288 | 128 × 256 | 1.000 | Original tile |
| MLP down | 9216 × 2560 | 128 × 128 | 1.136 | Original tile |
| GDN output | 4096 × 2560 | 128 × 128 | 1.057 | Original tile |

Both selected tiles use a 2×1 two-CTA cluster and 32-byte MMA K. These are
geometry changes in the pinned NVIDIA kernel, with no copied kernel body,
new quantization or BF16 fallback. The row policy is bounded at M32768 and
rejects unsupported rows. This sweep does not prove a global optimum or
measure Tensor Core, memory or stall counters.

## Serving and quality checks

`fp4_gemm_tuned.json` installs precompiled symbolic-M plans in the existing
shared projection-plan cache. The worker verifies native receipt sources and
GPU identity, reuses unchanged preparation receipts, and records the actual
tile for all four projections and both row bands. The benchmark rejects a
failed, stale-PID or incomplete runtime audit before timing. Loaded source
hashes and executed sources are retained separately from mutable working files.

The frozen training-seen systems-dev cohort remains 320 rows / 1,310,581 input
tokens with manifest SHA256
`b02af232d76935f2cda2a52213ebce27ae04ca5aaf44898cfe496104adad4266`.
Use 128 sequence slots, 32,768 scheduled/context tokens, 90% GPU memory,
disabled prefix caching and one constrained decision token. Complete fourteen
quick/full-sweep passes, an excluded full warmup, five c128 confirmations and
two further c1 confirmations. All 21 timed passes plus the excluded warmup
have matched IDs/prompts/token counts and finite scores/logprobs/margins.

Warm c128 rates are 180,740 / 183,330 / 182,682 / 182,902 / 181,867 input
tokens/s. Compare speed to the previous combined five-pass median 185,766.6.
For score comparisons, use its saved full-sweep c128 repeat-median scores,
whose macro/pooled AUROCs are 0.885192/0.908013. These differ slightly from
its earlier five-pass quality summary (0.884569/0.908326); keep the controls
explicit. Candidate score mean absolute difference is 0.002456. Artifacts
also retain per-source ranking, calibration, threshold and repeat diagnostics.
This cohort is used for systems development and establishes no final-ID
generalization claim.

Relative to the user-selected MXFP8 reference, throughput remains +5.03%,
while macro/pooled AUROC changes −1.6058/+0.7911 percentage points. The relative
HTTP score canary passes, while strict master/preparation failures inherited
from the combined stack remain failed. Passing the new tile checks does not
erase those precision receipts or promote this recipe.

## Artifacts and retained worker

- Native selection: `results/b200_attention_gdn_serving/fp4_gemm_tune01.json`
  and its exact five-source archive, including the pinned NVIDIA tile catalog.
- Serving sweep: `fp4_gemm_tuned_serving01`.
- Warm confirmations: `fp4_gemm_tuned_confirmation01` and
  `fp4_gemm_tuned_latency_confirmation01`; exact executed clients are collected.
- Collection: `fp4_gemm_tuning_collection01`, all 130 artifact hashes verified
  after transfer. Loaded kernel source and benchmark source hashes also match
  their executed archives.

The user subsequently selects this recipe as the reference. API **89879**, engine
**89962**, port **8010** remain warm and healthy on pod `i243nsg10usytq`.
Collection health returns HTTP 200, with the engine as the sole GPU process,
168,734 MiB used and idle temperature 33°C. `campaign.json` records completion,
configuration, receipts and artifact paths; no driver remains active. Stop this
server before changing kernels or starting another GPU worker. The selected
default now selects the tuned recipe; the FP32 master remains unchanged.


Selection accepts the measured latency/throughput tradeoff; it does not change
any result or reclassify the inherited strict precision failures. Future primary
throughput and warm-latency comparisons use the separate warm confirmations.
The latest collected kernel profile is the preceding combined-preparation run:
GEMMs total 32.00% (FP4 24.03%, BF16 7.97%), normalization/gates/layout 19.64%,
GDN core 14.37%, fused FP4 producers 13.00% and MXFP8 attention core 9.65%.
Because large-row tiles stay unchanged, this is a useful starting estimate for
the new reference, with no fresh post-tuning profile claimed.


## Output sampling cost, 2026-10-06

A read-only follow-up checks the pinned vLLM 0.24.0 sampler and the same saved
combined-preparation c128 trace. Requests use temperature zero, one generated
label, allowed tokens 15/16 and two processed logprobs. The sampler performs
argmax; random top-k/top-p sampling is skipped for all-greedy requests. The
`aten::topk` calls visible in the trace select returned logprobs, not random
samples. See the [pinned sampler source](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/sample/sampler.py).

Correlating GPU kernels by External id with their CPU operators gives 42
argmax kernels / 0.480 ms, 42 log-softmax kernels / 2.570 ms, and 840 top-k
helper kernels / 3.956 ms: **7.006 ms total, 0.1108% of 6.326 seconds of summed
GPU kernel time**. This core attribution excludes masking, rank/gather/copies
and host/API scheduling. CPU operator durations nest and are not wall-time
fractions, so this does not establish the complete output pipeline's overhead.

The last CPU `aten::mm` before each argmax maps to 42 inferred vocabulary-head
kernels / **8.284 ms, 0.1310%**. All grids have N dimension 1294 at tile N192,
consistent with the installed vocabulary of 248,320. Installed Qwen3.5
`compute_logits` calls the full-vocabulary logits processor; see its
[pinned projection source](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/logits_processor.py).
Keep this head attribution marked as inferred from ordering, grid and source.

For the monitoring score, `p(1) = sigmoid(logit_1 - logit_0)`: normalization
cancels in the logprob difference. A dedicated score endpoint could project
only the two decision rows and bypass token selection, full-vocabulary
log-softmax and top-logprob selection. Deleting the sampler outright from the
existing generation engine would break its output contract; retain a compatible
score interface. A smaller output GEMM can change floating-point rounding,
so adapter-specific score/AUROC parity remains required. The identified GPU
work totals only about 0.24%; no meaningful throughput gain is demonstrated,
and CPU savings require a matched measurement. No server/kernel change is made.

Receipt: `results/b200_attention_gdn_serving/output_sampling_diagnosis01.json`,
binding the existing trace hash, attribution method, operator counts and installed
source hashes. This is a low-priority simplification for the current long-prompt,
one-token workload, with potentially different economics for other workloads.


### Current cached EU host, 2026-10-07

`output_profile_euro01` checks the same hypothesis on the restored warm cached
server, without a restart, kernel change or baseline timing replay. Its saved
contract permits one engine trace on frozen quick64/c1 and full320/c128, stops
on input/response/health failure and forbids promotion or profiled speed claims.
Prefix caching stays off and chunked prefill stays on. The installed sampler,
projection and Qwen source hashes match the earlier diagnosis.

| Output GPU attribution | c1: 64 sampling steps | c128: 41 sampling steps |
|---|---:|---:|
| Inferred vocabulary head | 12.218 ms | 8.031 ms |
| Argmax, log-softmax and returned-logprob top-k | 9.050 ms | 6.653 ms |
| Allowed-token mask, gather and inferred rank reduction | 2.066 ms | 1.485 ms |
| Identified output work / all GPU kernel time | 1.562% | 0.277% |

At c1 the head/core median is **0.332 ms per request**, including a 0.191-ms
head; the extended attribution averages **0.365 ms per request**. This is the
scale of identifiable GPU work a two-logit path could remove, not a measured
end-to-end saving. That path still needs scoring and device-to-host delivery.
Engine dispatch, IPC, API response construction and serialization remain
unattributed: the profiler excludes the frontend, and nested CPU operator sums
cannot supply their latency. A complete score-endpoint bypass needs matched
unprofiled timing before promising additional host savings.

All head grids are `[1294,1,1]`, consistent with the installed full vocabulary;
rank reduction is inferred from the sampler's compiled greater-than count.
c1 scores match the archived same-host control exactly. One profiled c128 pass
has mean/max score differences 0.001925/0.086512, zero threshold flips and
macro/pooled AUROC deltas **-0.06556/-0.00586 percentage points**; this is
unchanged arithmetic under a different instrumented batch schedule, not a new
kernel-quality acceptance. Per-source, calibration, ties and undefined
single-label diagnostics remain in `score_comparison.json`. The archived warm
reference stays 30.36-ms c1 median and 211041 input tokens/s at c128; no fresh
speed result is claimed. Profiling is stopped and the sole server remains
healthy. Twenty collected files, including traces, executed client/analysis,
installed sources and closure, are checksum-bound under the run directory.

## GEMM follow-up research, 2026-10-06

The next low-cost screen is backend comparison on identical packed operands,
not a repetition of the same tile sweep. Installed FlashInfer 0.6.12 exposes
cuDNN, CUTLASS, TRT-LLM and CuTe-DSL FP4 runners; auto selection does not visit
all of them. Weight shuffles should be prepared once and cached. Our rowwise
inverse scales differ from the scalar alpha interface, so comparisons must
include the complete BF16-rounding/descaling path and any activation-layout
conversion. These are prospective trials, with no claimed speed gain.
See the [FP4 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
and [analogous B200 performance report](https://github.com/flashinfer-ai/flashinfer/issues/1732).

The larger fusion target is gate/up GEMM plus SwiGLU. Current fusion combines
SwiGLU with FP4 packing, leaving a BF16 gate/up tensor written by the GEMM and
read by the producer. The saved profile spends 10.4% on the fused SwiGLU
producer. NVIDIA's installed dense SM100 block-scaled SwiGLU kernel supports
NVFP4 inputs; [the GLU API](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/fe-oss-apis/gemm_fusions/grouped_gemm_glu.md)
uses alternating 32-column gate/up blocks and group-level alpha. BF16/FP16/
FP32/FP8 outputs are supported rather than our exact packed FP4 contract.
An initial integration could emit BF16 SwiGLU values, halving the intermediate
width before existing packing. Direct packed FP4 output requires preserving
per-row scaling before the nonlinear activation, both GEMM rounding boundaries
and the whole-row amax normalization. This involves adaptation; an unmodified
kernel is not a compatible replacement. Measure the complete producer path,
then warmed serving speed and AUROC; fewer launches alone are insufficient.

Layer-order attribution of the saved 1,722 BF16 matmuls suggests attention QKV
at 0.325 seconds (5.14% of total GPU kernel time), attention output at 0.131
seconds (2.07%), small GDN gates at 0.039 seconds (0.62%) and vocabulary head
at 0.008 seconds (0.13%). Each of 42 groups matches 24 GDN small gates, 16
attention projections and one checked vocabulary-head grid. Keep the family
labels marked as inferred from the frozen layer order, not profiler-recorded
module names. Large full-attention projection GEMMs are an additional FP8/FP4
candidate while the attention kernel and BF16 cache/recurrence stay fixed.
This precision intervention needs its own adapter-score/AUROC check; small
GDN gates are a much smaller target.

CUTLASS also uses cache-aware tile ordering, pipelining and split-K parallel
reductions; see [its GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).
Pinned FROST already implements L2 rasterization and derived stage sizing.
Counter-guided refinement is possible, but do not present these as missing
features. Split-K is primarily a small-M/high-K output-projection candidate;
large-M prefill already supplies many tiles and extra reduction traffic can
hurt. Hardware Tensor Core, DRAM/L2 and stall counters remain unmeasured.

`gemm_followup_research01.json` records source links, compatibility constraints
and the trace-bound BF16 attribution. Prioritize the bounded backend screen,
then adapted GEMM-plus-SwiGLU fusion. No new native/serving trial is launched,
no default changes, and the selected worker remains intact.


The user subsequently authorizes the backend comparison. All four alternatives
pass the matched native arithmetic cases, but no shape/row-band winner emerges.
Keep the selected FROST stack; the restored API/engine is now **91253/91317**.
See the [completed backend comparison](b200_fp4_backend_comparison.md) for
operator timings, the preserved/fixed CuTe API mismatch and artifact receipts.
