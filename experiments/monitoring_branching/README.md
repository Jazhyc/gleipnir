# Gradient-preserving prompted branches

Hypothesis: sharing the causal trajectory computation across all intermediate
decision suffixes can reproduce independent-prefix training while reducing
repeated forward/backward work. This is a systems change, not an auxiliary
classifier: every branch retains the exact existing prompted decision interface.

Baseline: independently forward each complete tokenized prefix prompt plus the
full endpoint; use the same normalized soft BCE losses and targets. Intervention:
process shared trajectory segments once, fork differentiable KV/conv/recurrent
states at boundaries, and process each original suffix independently. Never feed
a branch's closing tags or prediction tokens into the continuing trajectory.

Token sharing uses actual longest common token prefixes, not character offsets;
BPE joins can change the last trajectory token when the suffix is appended.
Avoid one-token cached continuations because they invoke inference-specific
in-place convolution/recurrent kernels. Move a split earlier when necessary,
without changing any request's token sequence.

First gate: tiny mixed full/linear-attention Qwen CPU model in deterministic FP32,
eager attention, no dropout, no checkpointing. Compare decision logits, the
normalized objective, and every parameter gradient against independent requests.
Assert branch-state isolation and fewer processed tokens. This reference check
intentionally uses the Torch fallback and is not a production training recipe.

Second gate (pending idle GPU capacity): real pinned FLA/causal-conv kernels,
BF16 and rank-128 QLoRA; compare scores, gradients and a matched optimizer update
before timing representative and longest trajectories. Require finite gradients,
nonzero gradient through shared history, preserved full Kimi targets, no leakage,
and matched parent normalization. No ID-based tuning of systems tolerances.
Do not switch a running campaign or claim throughput gains before this gate.

The prototype fails closed on checkpointing and dropout. Differentiable caches
retain activations: all-boundary memory may exceed the current sampled-prefix
recipe. Functional checkpointing or a layerwise implementation may be needed;
ordinary inference-cache reuse or detached states is not an exact substitute.

Run the bounded CPU correctness entrypoint:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.monitoring_branching.run
```

Current prefix-weight runs are untouched. No GPU benchmark is automatically
queued, and no monitoring of those runs is resumed by this experiment.

## Initial CPU result

The mixed-attention FP32 canary passed on 2026-09-06: maximum decision-logit
error 1.49e-8 and maximum parameter-gradient error 2.38e-7. The four exact
requests processed 40 token positions with sharing versus 75 independently.
This is correctness evidence for the eager reference path only, not a measured
H100 speedup or proof that FLA final-state gradients and QLoRA match. Full-scale
memory optimization, production-kernel parity, and trainer integration remain
pending. Do not replace the active sampled-prefix runs with this prototype.
