# Adaptive B200 physical microbatches

Date: 2026-10-01. Status: implementation tested; first GPU correctness preflight
failed, diagnosis in progress. No adaptive timing result or recipe promotion.

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
