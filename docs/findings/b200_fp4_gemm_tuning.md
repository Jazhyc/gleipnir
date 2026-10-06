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
