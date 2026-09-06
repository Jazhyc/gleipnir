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

## Authorized all-prefix experiment

User now requests one all-prefix run, weight 0.1, using both GPUs for training.
Preserve LR 2e-5, one epoch, seed 0, all 8,688 parents and original Kimi
endpoint targets. For each eligible parent use
`(L_full + 0.1 * sum(L_prefix)/K)/1.1`; prefix-free parents retain L_full.
Include all 133,947 approved cached prefix boundaries, retaining the previously
authorized numerical-cache exception; do not relabel or rescale targets.
Compare final ID evaluation with full-only and one-sampled-prefix weight 0.1;
no OOD selection or automatic extra hyperparameter search. Do not claim an
all-prefix benefit based on numerical-kernel canaries.

Before trainer integration, run a bounded real 4B QLoRA parity diagnostic on
GPU 0 using the existing trained 0.1 adapter to exercise nonzero A/B gradients.
Use identical synthetic requests of length 259/516/1027, production FLA and
causal-conv1d, BF16, eager SDPA and zero dropout. Freeze maximum margin error
0.1, probability error 0.02, global gradient relative L2 0.05 and cosine 0.995.
Stop on missing/nonfinite gradients or a failed gate. This short gate is not
sufficient to launch training: optimizer-step, long-sequence memory, functional
checkpointing and two-GPU gradient/normalization checks remain required.

## Production preflight progress (not a training launch)

Added on-demand exact all-prefix request reconstruction in
`gleipnir.branch_data.BranchDataset`, preserving cache/parent/target checksums,
and opt-in functional segment checkpointing using fresh cache containers during
both forward and recomputation. The maximum prefix count is 99 (median 17 among
6,348 eligible parents); all 133,947 targets remain in scope. GPU canaries load
the trained weight-0.1 adapter solely to exercise nonzero A/B gradients; an
eventual matched full training run must initialize fresh adapters from the base.

The short synthetic real-4B production gate passed with identical scores,
gradient relative L2 0.017312, cosine 0.999850. Checkpointing reproduced that
pass (relative L2 0.017320). But real parent 0 failed: margin error 0.125,
probability error 0.001445, gradient relative L2 0.246704, cosine 0.992256.
Alignment to 64-token boundaries preserved exact requests but still failed:
relative L2 0.225525, cosine 0.996498. Do not relax the frozen gates or call
small probability error evidence of correct training gradients.

Longest-token parent 8142 also has 99 prefixes. Single-GPU checkpointed execution
ran out of memory. Explicit two-GPU layer placement (layers 0–15/GPU0,
16–31/GPU1, tied input/output embeddings both GPU0) completed a bounded
forward/backward/AdamW update in 123.28 seconds. Peak allocations were
67,653,889,024 and 64,458,370,048 bytes. This establishes one-case memory
feasibility, not full-workload correctness or a mean training ETA. Data
parallelism would replicate the overflowing per-trajectory state, so the
proposed two-GPU training arrangement is model parallel, with accumulation 32.

FP32 selected-token head projection on the two-GPU aligned path reduced real
parent-0 gradient relative L2 to 0.081498, still failing 0.05. A further optional
variant computes the full Kimi endpoint independently and shares only the prefix
trunk. Its first GPU check exposed an embedding-gradient hook left behind by
Transformers' public checkpoint-enable method. The implementation now toggles
layer checkpoint flags without installing that hook; a frozen-embedding CPU
regression covers the issue. Eighteen focused CPU tests pass across checkpoint,
FP32-head, independent-endpoint, cache isolation, exact requests and prefix loss.
The corrected hybrid GPU gate remains pending; no all-prefix run is launched.

Diagnostics and failures are retained under `results/monitoring_branching/` and
`logs/lambda/monitoring_branching/`. Optional precision/alignment/endpoint variants
are systems diagnostics, not teacher-target calibration. No annotation changes,
threshold loosening, OOD tuning, or long-run launch follows from a failed gate.
