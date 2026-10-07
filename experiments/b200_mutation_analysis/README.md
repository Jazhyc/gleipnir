# Qwen RoPE mutation analysis

Hypothesis: Torch 2.11's mutation analysis misclassifies six runtime stride
scalars as constants when constructing TTIR for the pinned Triton 3.7.1 kernel.
Its conservative all-inputs-mutated fallback may introduce avoidable copies or
restrict optimization. Reproduce the failure, then correct the TTIR constant
map and runtime argument positions, filtering scalar mutation results after
tracing. Apply only to the checksum-bound Qwen kernel. Preserve its source,
runtime launch and arithmetic; do not suppress warnings or hard-code mutations.
The stride-constexpr prototype is unadopted and preserved separately. Initial
native harnesses make model head geometry symbolic and fail in the unmodified
control too. The corrected harness keeps that geometry constant, matching serving,
while tensor sizes and strides stay dynamic. No runtime upgrade is involved.

Before GPU trials, retire the sole identity-verified score server. Native checks
must show the original analysis failure, corrected TTIR mutation results containing
only Q/K/gate outputs, exact eager/compiled agreement, unchanged inputs, strided
layouts, finite outputs and changed-input CUDA-graph replay. Compare bounded native
timings and profile copy operators; exclude profiling from speed claims. Bind
original installed-source, generated kernel and helper hashes. Stop on failure,
preserve receipts, and restore a useful server.

Integrate into the same cached two-logit score server only after the native gate.
Reuse its `score02` controls and the selected cached generation controls; no
control replay. Keep model/adapter, native Gigatoken/direct FROST, FP4 scope,
MXFP8 attention, BF16 state, causal LAST pooling, KV cache, chunked prefill and
prefix-cache-off policy unchanged. Use one excluded warmup plus three c1/quick64
and six c128/full320 repeats, exact token counts and every score. Report input
tokens/s, latency bins, pooled/per-source/source-macro AUROC, calibration, ties
and threshold diagnostics. The frozen cohort is training-seen systems development;
no final-ID use or automatic precision/reference promotion.

Stop on source/input drift, nonfinite or mismatched native output, unexpected
mutation sets, score-canary failure, server failure, or suite completion. A warning
fix is not itself proof of a speed gain. Preserve inherited strict-master failures
separately from passing accepted-reference checks. Keep the final useful server
warm, its logs and shared compiler caches; no capacity lifecycle change.

```bash
PYTHONPATH=src:. python -m experiments.b200_mutation_analysis.run --name mutation01
```

See [the upstream issue](https://github.com/pytorch/pytorch/issues/170049) for the
runtime-scalar/constexpr argument-index mismatch. Only the checksum-bound kernel
is supported; different Torch/Triton versions or upstream source need new validation.
