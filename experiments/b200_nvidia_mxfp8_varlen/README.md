# Direct packed NVIDIA MXFP8 attention

Hypothesis: one native variable-length causal D256 GQA attention call per
physical row removes the dense prototype's per-example dispatch and exact-length
compilation overhead, improving complete LoRA update time. Keep BF16 boundaries,
FP32 master adapters, BF16 MLPs and all 24 pinned FlashQLA GDN layers unchanged.
This is experimental; the standard recipe remains BF16 FlashAttention 4.

Implement packed THD payloads, sequence-local 32-element quantization blocks,
packed forward scale tiles and native backward cumulative-length arguments.
Runtime token totals, cumulative lengths and maximum length should reuse one
compiled artifact for each fixed head/layout contract. Include quantization,
scale repacking, scratch allocation and backward in timing; do not hide these
costs in preparation or use CPU per-example attention calls.

Before training, compare mixed lengths around 32- and 128-token boundaries with
the original dense MXFP8 implementation and independent per-example FP32 causal
attention. Check all quantized payloads and scale bytes against the pinned NVIDIA
quantizer, singleton behavior, finite gradients and cross-example isolation.
Exercise changing lengths under reused compiled plans and CUDA graph capture.
Stop on unsupported layouts, missing/nonfinite gradients, isolation failure,
memory exhaustion or dependency drift. Preserve failed numerical receipts.
Strict forward/gradient limits remain 2%/5%; the prior 10% learning ceiling and
explicit timing-only exception must be reported separately.

Only after native execution checks, run fresh whole-model eager/compiled packing
and largest-row memory checks, then a bounded 20-update screen (ten warmup, ten
measured) from the frozen initial adapter and input contract. Historical matched
controls are dense MXFP8 5.29781 s/update and BF16 FA4 4.08648 s/update; keep their
recorded runtime difference explicit. No held-out quality promotion follows this
systems experiment. Recommend further work only with at least 5% improvement in
complete update time; retain the negative result otherwise.

Outputs: `results/b200_nvidia_mxfp8_varlen/`; logs:
`logs/runpod/b200_nvidia_mxfp8_varlen/`. Use the existing authorized NC2 B200 and
shared persistent compiler caches. This session has no scheduling tool; monitor
startup and bounded execution during the active turn without promising later
agent heartbeats.

Native attempt `native05` completes the fresh producer-layout, exact dense
MXFP8 comparison, singleton, isolation, changed-length graph-replay and poisoned
scratch checks. Independent FP32 strict parity still fails. The bounded
`training01` screen completes 20 finite updates under the earlier explicit
timing-only authority. It averages 3.93793 s/update versus 5.29781 dense MXFP8
and 4.08648 historical FA4, with identical physical contracts and a byte-for-byte
identical final FP32 adapter to the dense MXFP8 run. Whole-model eager/compiled
gradient parity still fails at 18.7114%/16.6558%; keep FA4 as the standard. See the
[finding](../../docs/findings/b200_nvidia_mxfp8_varlen.md) for receipts and limits.

```bash
python -m experiments.b200_nvidia_mxfp8_varlen.run --attempt native06
python -m experiments.b200_nvidia_mxfp8_varlen.training_screen
```

Launchers reject existing output directories so previous receipts survive.

## Attention bottleneck diagnostic

Hypothesis: the small complete-update gain reflects both work outside full
attention and FP8 conversion/backward costs. Compare warmed BF16 FA4 and the
unchanged validated packed MXFP8 forward/backward on identical BF16 operands,
including all quantization, repacking, scratch initialization and gradients.
Use controlled short, balanced and skewed packs plus an actual long-singleton
length from the measured training window. These synthetic operands are a kernel
diagnostic, not a new complete-model throughput or quality result.

`profile_attention.py` performs six warmups and ten timed repetitions per
backend/shape, alternates backend order across shapes, measures CUDA graph
replay separately, and traces one warmed iteration. It reuses existing native
validation and persistent caches, makes no optimizer updates, and stops on
nonfinite outputs/gradients, unsupported execution or OOM. Record CPU wall and
CUDA-event times separately; profiler sums are attribution, not throughput.
Stop after four shapes, with no tuning sweep or new numerical canaries.

Completed `profile01`: long-singleton attention improves 17.926 to 12.903 ms,
but short-pack attention regresses 0.819 to 5.181 ms. Graph replay removes much
of the short-input host cost; 38–39 MXFP8 GPU kernels remain versus four for FA4.
The long-input dQ kernel is roughly unchanged at 4.9 ms; forward and dK/dV
improve. This is attention-only evidence, not a measured whole-model component
breakdown. See the finding for actual-kernel attribution and trace hashes.
