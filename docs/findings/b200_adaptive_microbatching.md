# Adaptive B200 physical microbatches

Subsequent decision (2026-10-01): the user explicitly selected compiled
16,384-token/max-8 adaptive batching for future training. See the
[decision record](../decisions/b200_adaptive_training_recipe.md). This choice
does not change the historical parity failures and diagnostic decisions below.

Date: 2026-10-01. Status: implementation tested; bounded batching, singleton
repeatability, matched-loss, compiler-autocast and actual-update/learning-curve
audits completed. No adaptive candidate passes the original gradient gate;
no timing-screen result or recipe promotion. Root cause remains unresolved.
The separately authorized [execution audit](b200_execution_audit.md) finds
close ten-step mean losses despite different gradients and updates.

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

## Matched eager/compiled forward losses

Source `94176ac799ba0eed9bf55b757227cfeeec202901` ran the enhanced compiled
eight-input canary against the same frozen job and seed, without optimizer
updates. The earlier eager diagnostic provides the matched forward reference.

| Physical batching | Eager mean loss | Compiled mean loss | Compiled relative difference |
| --- | ---: | ---: | ---: |
| Eight accumulated singletons | 1.066599831 | 1.061895311 | -0.4411% |
| One padded batch of eight | 1.068278551 | 1.065031767 | -0.3039% |

Maximum decision-logit and decision-margin differences are both 0.0625 for
each batching mode. Maximum absolute binary-probability gaps are 0.014056 for
singletons and 0.011747 for batch 8. Individual singleton losses can differ by
up to 0.050507, so close averages alone do not establish per-example identity.
These are forward diagnostics on eight truncated training inputs, not a
learning-curve or held-out-quality result.

The new within-compiled singleton-versus-batch-8 gradient check fails:
relative L2 **0.753615**, cosine **0.659332**, norms **32.500166** and
**23.102878**. Preserve this separately from the original 0.519511 failure;
it does not constitute a controlled identical-shape repeatability test.
The relative loss agreement cannot validate the corresponding backward path.
We have not directly compared eager and compiled gradient vectors at fixed
physical batching. The gradient norm difference is a diagnostic clue, not
such a cross-backend parity measurement.

Artifacts: `results/b200_adaptive_microbatching_diagnostic_compiled_loss/`,
including the derived `eager_compiled_comparison.json`, and
`logs/runpod/b200_adaptive_microbatching/diagnostic_compiled_loss.log`.

## Scoped compiler backward-autocast audit

The pinned Torch source defaults to `backward_pass_autocast="same_as_forward"`,
while the canary applies BF16 forward autocast and calls backward outside it.
[PyTorch's documented compiled-autograd semantics](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_backward.html)
prescribe `"off"` for that pattern and warn that a mismatched assumption can
silently affect correctness. Source `2822fbabd1efdec313ed6c168e0a29f6465cab0c`
passed 25 focused tests, including setting/argument restoration after success
and failure. An isolated subprocess then reran the compiled eight-input canary
with that override and unchanged original job. Ordinary training was unchanged.

The override **does not resolve parity**: relative gradient L2 **0.647156**,
cosine **0.764318**, reference/actual norms **32.495305**/**23.059149**.
Singleton loss remains exactly **1.0618953108787537**; batch-8 loss is
**1.058821678161621**. That batched forward change means this is not empirical
proof that only backward numerics changed. Without controlled repeats, do not
attribute the smaller discrepancy solely to the setting. All results remain
below the required level of gradient agreement; no optimizer update occurred.

Artifacts: `results/b200_adaptive_microbatching_diagnostic_backward_off/` and
`logs/runpod/b200_adaptive_microbatching/diagnostic_backward_off.log`. Both new
diagnostics were collected locally. The GPU is idle; the live-read Pod remains
running at $6.79/hour as requested.

The next practical checks are fixed-physical-batch eager/compiled gradient
comparison, actual clipped AdamW update direction and magnitude, and post-update
loss on a common execution path. A separately predeclared bounded matched
learning-curve diagnostic would test whether differences materially affect
training. Equal starting loss does not establish equal updates, and gradient
L2 alone does not establish worse final quality. Do not silently relax the
original parity gate or launch its failed timing campaign. No full campaign is
validated by these forward checks; compile and batching numerical behavior
remain unresolved.

## Completed actual-update and learning-curve audit

The user subsequently authorized all three remaining checks, including optimizer
updates despite the failed gradient screen. Source
`5767c3b56eb57519ef6ecc4de2beaf5a1ee28f30` passed 49 focused CPU tests and
completed four gradient/update probes plus four matched ten-step trajectories.
Fixed-partition eager/compiled gradient relative L2 is 0.102–0.111 and actual
clipped AdamW update error is 0.324–0.337. Final mean losses on a common eager
training-probe path span 0.927562–0.940055, a 1.35% spread. There were no OOMs
or unexpected compiler fallbacks.

These probes select eight rows from the matched 320-row cohort, unlike the
historical global-longest-eight stress inputs above. They do not invalidate
the earlier failures. Adaptive policies realized physical sizes 1, 2, 4 and 8;
their step-compute timings are promising but are not a validated complete-loop
benchmark or revised ETA. See the [execution findings](b200_execution_audit.md)
for all comparisons, per-example limits, provenance and artifacts. Keep the
original 0.05 gate and selected fixed-batch recipe; no held-out quality or full
training campaign was run.
