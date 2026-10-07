# B200 FROST host binding wrappers

2026-10-07. Retain the direct host binding path on the native Gigatoken/Direct
FP4 worker. Same-worker comparisons give a **0.60% median paired throughput
gain**, with five of six pairs faster, and c1 median/p95 latency improves from
156.84/190.96 to 149.69/184.19 ms. Native outputs are bitwise equal. This is a
small measured systems gain, not a new GPU precision or quality recipe.

## Intervention and validation

The cuDNN variant-pack resolver rebuilds tensor-object/name/UID maps on every
call. The serving wrapper resolves its fixed six-tensor operand positions once
and calls the same pinned compiler's `lowered` executor directly. Keep its
device, shape, layout, alignment and SF guards, compiled kernel, tile selection,
row descale and BF16 rounding. Cache frozen weight views with tensor identity,
pointer, shape, stride, dtype and device in the key; bounded strong references
prevent stale views after storage/layout replacement. The cache stores views,
not weight values. Scale and returned output tensors remain freshly allocated;
mutable activation/output pooling is excluded from this intervention.

Stop the old sole API/engine before GPU checks. The focused screen covers all
five unique projection geometries at rows 1/129/4096/32768 with the already
selected small/large tiles. All 20 cases are finite and bitwise equal to the
original call, including independent outputs, changed inputs, changed-weight
CUDA-graph replay and a second CUDA stream. It reuses existing native kernels
and their numerical receipts. Small-row host-inclusive synchronized native
calls improve by roughly 18--29%; these microbenchmarks exclude packing and
are diagnostic, not the serving speedup.

Preserve two failed setup/test receipts. `host_wrapper_canary01` omitted the
existing FA4/CUTLASS 4.8 overlay and failed importing `cutlass.experimental`.
`host_wrapper_canary02` used unsupported in-place FP8 multiplication in its
changed-weight fixture. Correct the launcher and update FP8 through FP32 plus
copy-back; `host_wrapper_canary03` passes. Neither failure was resolved by
changing kernel arithmetic or widening a numerical tolerance.

The optional initializer patches only the resident serving class's host call
and adds a loopback development RPC for matched mode switching. Its helper and
validation hashes are recorded separately from GPU configuration. Reuse the
existing GPU compile identity and AOT/cache entries. The replacement server
`host_wrapper_start01` is ready in 97.04 s and passes the twenty-row serving
score canary. Original CPU affinity and native Gigatoken remain fixed.

## Warm serving results

Three alternating c1/quick64 pairs and six balanced c128/full320 pairs execute
on the same API/engine, with excluded warmups. Keep ordinary text requests,
data/order, model, caches and GPU settings unchanged. Verify mode-specific
callback increments and process identity around each pass. Rows are the frozen
training-seen systems-development examples; final ID is excluded.

| Measurement | Original binding | Direct binding |
| --- | ---: | ---: |
| c1 median latency | 156.84 ms | 149.69 ms |
| c1 p95 latency | 190.96 ms | 184.19 ms |
| c1 median input tokens/s | 28273 | 29399 |
| c128 median input tokens/s | 196558 | 197530 |
| c128 median latency | 2.40324 s | 2.40214 s |
| c128 p95 latency | 2.83723 s | 2.83419 s |

Median c128 paired gain is +0.60%, versus +0.49% for the ratio of median rates.
Pair changes are -2.94/+0.59/+1.35/+0.61/+0.27/+0.71%. Preserve the first
negative pair; a six-pair bootstrap interval is -1.33% to +1.03% and includes
no change. The candidate meets the frozen >0.5% paired-median, >=5/6 positive
pairs, <=2% c1 regression rule and quality guards, but this is not a guaranteed
production improvement. The native call saving largely overlaps GPU work at
high concurrency. c1 median/p95 improvements are 4.56%/3.54%; only three latency
repetitions are available.

c1 scores/margins are exact, with no threshold flips or AUROC change. c128
source-macro/pooled AUROC changes by **+0.0426/+0.0391 percentage points**, with
zero flips. Mean/max score difference is 0.000913/0.056559; margin difference is
0.013086/0.375000. Keep scheduling-dependent FP4 variation and all inherited
strict precision failures separately; bitwise native parity does not imply
equal scores under differently formed GPU batches or improved generalization.

## Engine profile and limits

Separate original/direct engine traces are excluded from speed measurements.
Both execute 41 nonempty batches and 39 full 32768-token batches, totaling
1310581 input tokens. Summed kernel time is 5.83743/5.78188 s and the GPU
window is 6.44739/6.30986 s. Before-next-launch gaps excluding copies fall from
480.92 to 397.29 ms. This supports the direction of the intervention but does
not establish a steady 83.64-ms host saving.

The first nonempty context changes from 561 to 245 tokens; its gap overlap
falls 283.97 to 183.16 ms, more than the 81.95-ms reduction in total gaps.
The 245-token batch also fits the existing 256-token graph envelope, while 561
does not. Original is the first profiler activation. Shapes/graph eligibility
and profiler initialization confound the difference. Kernel counts are
43688/43692. Do not attribute these GPU-work or gap differences entirely to
host binding reuse or use profiled HTTP timing as the measured speedup.

## Artifacts and retained worker

Artifacts under `results/b200_attention_gdn_serving/`: all three native checks,
`host_wrapper_start01`, `host_wrapper_compare01` (predictions, paired traces,
executed clients/profile analyzer), and `host_wrapper_collection01` (logs,
retired-server receipt, final runtime/campaign metadata and checksums). All
**126 collected files** verify locally. Thirty-five focused tests and Ruff
pass. Source and compiler bindings remain explicit; no model weights are copied.

API **109400** and sole GPU engine **109438** remain healthy and warm on port
8010, using direct host bindings and native Gigatoken. Original affinity is
retained, profiling is stopped, and GPU memory is 169116 MiB. The selected GPU
baseline checksum and compile identity remain unchanged. Future compatible
trials can reuse this worker. To reconstruct the same host setup after retiring
the previous API/engine:

```bash
PYTHONPATH=src:. python -m experiments.b200_attention_gdn_serving.startup --name unique_wrapper_start --frontend-validation results/b200_inference_benchmark/gigatoken_native_canary03/validation.json --frontend-ab --host-wrapper-validation results/b200_attention_gdn_serving/host_wrapper_canary03/validation.json
```

The control is opt-in; original bindings remain available. New helper/compiler
sources require renewed admission, rather than silently accepting old receipts.
