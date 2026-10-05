# Fused MXFP8 operand preparation

Hypothesis: adopting the Meta blog's single-pass production of FP8 operands and
consumer scale layouts reduces the packed NVIDIA path's conversion and dispatch
cost. Test arithmetic-preserving dual row/column quantization first, then Meta's
transpose-invariant 32x32 block scaling. Retain causal D256 GQA, BF16 producer
boundaries and gradients, all 24 FlashQLA GDN layers, BF16 MLPs and FP32 master
adapters. The BF16 FA4 default and existing MXFP8 controls remain available.

Intervention: one sequence-local preparation kernel per Q/K/V/dO operand emits
the compact payload and native forward/backward scale layouts. Square scaling
shares one FP8 payload between both contraction orientations. This implements
the producer/layout fusion principle at the attention boundary, not a claim to
have ported Meta's RMSNorm/GEMM epilogues or its unsupported attention kernel.
Qwen's Q/K normalization and rotary embeddings precede this boundary. Meta's
FP16 global dQ reduction is not applicable directly to NVIDIA's separate dQ
kernel, which retains its accumulator locally and writes the final gradient.

Before timing, compare dual payloads and every live/native scale byte with the
existing packed producer and repacker at mixed 32/128-token boundaries. Compare
outputs and input gradients exactly for dual mode; compare square mode with an
independent FP32 causal reference and retain strict failures. Test singleton
semantics, sequence isolation, changed-cut graph replay and poisoned allocations.
Stop on nonfinite results, missing gradients, layout/isolation failures or OOM.
As a separate project optimization, fuse the cumulative-boundary predicates
into one device kernel followed by the existing asynchronous CUDA assertion;
preserve the host envelope and reject invalid cuts before operand preparation.
Square quantization may couple positions within a 32-token block; retain the
existing across-block future perturbation check and report within-block effects.

Then compare four frozen diagnostic shapes with both old MXFP8 and BF16 FA4,
including all preparation and backward work. Six warmups, ten uncaptured
repetitions and twenty graph replays; no tuning sweep or optimizer updates.
Proceed to a bounded 20-update matched training screen only after native
execution checks, fresh eager/compiled packing checks and longest-row preflight.
Keep the earlier explicit timing-only authority and strict 5% / separate 10%
gradient limits. Use identical initial adapter, cohort, seed and packing to the
direct-varlen controls. Stop at 20 updates; no held-out quality promotion.
Require at least 5% complete-update improvement to recommend the intervention.

Use the existing authorized NC2 B200 and persistent compiler caches. Outputs:
`results/b200_mxfp8_fused/`; logs: `logs/runpod/b200_mxfp8_fused/`. Active-turn
monitoring checks startup and bounded runs; no later heartbeat is promised.

Source: [Meta blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
sections 2.2 and 2.3, public source revision
`2889aefba0d03b3eecf779eed9f4097066ecdc5d`. The fused kernels below are a project
implementation of those ideas using NVIDIA's pinned consumer layouts, not
copied unpublished Meta producer code.
