# Adaptive B200 physical microbatches

Date: 2026-10-01. Status: implementation tested; bounded batching probes and
singleton repeatability audit completed. No passing adaptive candidate,
adaptive timing result, or recipe promotion. Root cause remains unresolved.

The user authorized empirical optimization with ten-update workloads on the
existing B200. The [experiment contract](../../experiments/b200_adaptive_microbatching/README.md)
preserves 32 equally weighted traces per optimizer update, sorting only within
that logical batch. Candidate padded-token budgets are 8,192/max batch 4 and
16,384/max batch 8. The control uses the same logical Trainer path with physical
singletons. All retain the validated twelve-checkpoint SDPA/FLA recipe.

Source `5e4b32ba6340c2a1b1e280f2289a45c0851b6153` passed 53 focused CPU tests
on the Pod. Tests exercise real Trainer/Accelerate optimizer boundaries,
unequal microbatch weighting, partial final batches, analytic gradients,
coverage/identity checks, token budgets, and nonfinite-loss rejection. The
earlier local snapshot also passed its 51 tests.

Preparation reproduced matched cohort SHA-256
`a27b8b3d702436839869fda2aa612ad61a4c3ef8aa9f56b4c57a4914b0a50e06`
and longest-32 SHA-256
`1ed6774f5234b606a52b19d10a93bf0da9bfbad69b3ffe67af8ef8608e4ed875`.
Data, seed, targets, Qwen revision, FP32 rank-128 adapters, NF4/BF16 quantization
and checkpoint policy match the preceding systems screens.

The pinned kernel and eager/compiled logit checks passed. The same-weights
gradient check over eight truncated inputs of lengths
`[2048, 2048, 2048, 2048, 1024, 512, 256, 128]` failed when combined into one
padded physical batch of eight. It compares all 169,869,312 trainable parameters:
reference gradient norm 28.41117, relative L2 error **0.519511**, maximum absolute
error **0.607147**, versus the predeclared relative tolerance **0.05**. Memory
around 82 GiB was observed before the failure; this is not a completed peak-memory
benchmark. The process stopped before any optimizer update or timing condition.

Retain `results/b200_adaptive_microbatching/` and
`logs/runpod/b200_adaptive_microbatching/` as failed diagnostic artifacts. The
CPU normalization result does not establish model/kernel padding parity.
Compare eager and compiled paths with the same eight inputs and record logits,
losses, gradient norms/cosine and parameter differences before changing any
policy. Keep the existing B200 recipe and its provisional 3.52-hour epoch compute
estimate until matched measurements justify a change.

## Eager diagnostic

Source `ffd3e1f29e3079e4681730bd4f002ffd8699aa6d` passed 21 focused tests of the
enhanced diagnostics and ran the same canary without compilation. It still
fails the 0.05 relative gradient threshold: error **0.137635**, cosine **0.990498**,
reference norm **23.42287**, actual norm **23.07239**, maximum absolute error
**0.059845**. Reference and batched mean losses are **1.066600** and **1.068279**
(0.157% difference). Decision-logit differences reach 0.0625, with a margin
difference up to 0.125. Attention dropout is zero in the frozen backbone config.
This does not establish a padding bug or justify loosening the gate; BF16
execution and compilation sensitivity need to be separated. The largest gradient
differences occur in early full-attention and MLP adapters.

Artifacts: `results/b200_adaptive_microbatching_diagnostic_eager/` and
`logs/runpod/b200_adaptive_microbatching/diagnostic_eager.log`. No optimizer
updates occurred. The next bounded diagnostic isolates LM-head precision and
right-padding mask handling; all variants retain the same eight inputs and
weights and restore the original model methods afterward. The same diagnostic
also checks smaller physical maxima 4 and 2, explicitly recording each policy
instead of silently reducing the original benchmark's batch size.

## Completed bounded probes and decision

Source `1af8a04e9c8b9614031265272e2d7ce6455c578f` passed 22 focused diagnostic
tests, including scoped restoration of model methods and FP32 projection under
outer BF16 autocast. The same eager model then ran all six probes without any
optimizer update. The repeated baseline reproduced the preceding result:
relative L2 error 0.137673, cosine 0.990492. All probes retain the 0.05 gate.

| Physical maximum | Projection | Right-padding mask | Gradient relative L2 | Cosine |
| --- | --- | --- | ---: | ---: |
| 8 | FP32 head | Preserved | 0.139650 | 0.990205 |
| 8 | Original BF16 head | Omitted in diagnostic | 0.092214 | 0.995745 |
| 8 | FP32 head | Omitted in diagnostic | 0.093845 | 0.995598 |
| 4 | Original BF16 head | Preserved | 0.133771 | 0.991012 |
| 2 | Original BF16 head | Preserved | 0.134385 | 0.990930 |
| 2 | FP32 head | Omitted in diagnostic | 0.093331 | 0.995643 |

Reducing physical size alone does not meet the gradient threshold. FP32 output
projection does not resolve the discrepancy. Omitting the causal right-padding
mask reduces it, but still does not pass. This is evidence of sensitivity to
execution details, not a demonstrated upstream bug or an accumulation-weighting
error. The normalization tests establish the latter independently on CPU.
No precision/mask probe is enabled for ordinary training or recommended as a
recipe. The eight-input tail-truncation stress test is also distinct from a
model-quality evaluation; no held-out scores were consulted.

The scientific stop condition was reached: **keep the validated fixed batch-1
B200 recipe**. The implementation and profiling records remain opt-in and
experimental. The ten-update conditions and longest-32 optimizer update did not
run because the gradient canary precedes them. There is no adaptive throughput
estimate and no evidence for a revised epoch ETA. The previous provisional
3.52-hour compute estimate remains unchanged; setup/export are additional.

Artifacts and complete decision-logit/gradient diagnostics were collected locally
from `results/b200_adaptive_microbatching_diagnostic_probes/`; all failed screen
and diagnostic logs remain under `logs/runpod/b200_adaptive_microbatching/`.
The GPU is idle after the bounded probes. Pod `alzfug70g5237b` is left running
as requested at $6.79/hour; no additional capacity was provisioned.

## Identical-shape repeatability and kernel audit

Source `43361ba34fdf4a7ddc151f7205ce2790126751a1` passed 23 focused tests and
ran the same eager eight-input canary with maximum physical size 1. Reference
and actual passes now use identical singleton shapes, ordering and weighting.
Relative gradient L2 error **0.005119** passes the unchanged 0.05 gate; cosine
is **0.999987**, maximum absolute error **0.002380**, and both mean losses are
exactly **1.0665998309850693**. Reference/actual gradient norms are 23.422508
and 23.424437. No optimizer updates occurred. Collected artifacts are in
`results/b200_adaptive_microbatching_diagnostic_repeatability/` with
`logs/runpod/b200_adaptive_microbatching/diagnostic_repeatability.log`.

The cross-shape eager discrepancy is roughly 27 times this repeatability error.
This rules out singleton repeat-to-repeat variation as the sole explanation for
the larger difference. It does not test repeatability of batch 8 or the compiled
path, prove a padding bug, or identify which operator causes the discrepancy.
Loss weighting is mathematically partition-invariant and independently tested;
BF16/quantized shape-dependent kernels and compiled execution remain plausible
causes. Relative gradient L2 is not relative loss error or a quality metric.

The installed pinned FLA 0.5.2 source includes the Blackwell forward-state
two-warp guard from [FLA PR 953](https://github.com/fla-org/flash-linear-attention/pull/953).
A later [output-kernel race report](https://github.com/fla-org/flash-linear-attention/issues/1228)
concerns GB300/sm103, Triton 3.6/3.7 and a different model/load. Our output-kernel
source still contains an eight-warp autotune entry, but the report does not
establish that our B200/sm100 run is affected. No workaround or dependency upgrade
was applied. `kernel_source_audit.json` records installed source hashes and guards.
The fixed batch-1 recipe remains selected, and the Pod is idle and left running
at the live-verified $6.79/hour.
