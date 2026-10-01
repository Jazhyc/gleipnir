# FP4 stability diagnostic

Hypothesis: the previous eager/compiled loss disagreement must be separated
from low-precision backward instability. First reproduce forward comparisons
with original BF16 MLP bases and native Four Over Six bases, repeated calls,
and prefixes of the existing compiled decoder policy. No optimizer updates in
the diagnostic stage. Keep the existing absolute 0.01 + 1% loss gate.

Then test BF16 input-gradient GEMMs using the BF16-dequantized exact forward
packed weight. Frozen bases do not need weight gradients; LoRA branches retain
FP32 master adapters. This adopts the dequantized-backward idea from
[The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl), not its MoE/RL stack.
Native W4A4 forward GEMMs remain explicit CUTLASS, with MSE 4/6 block selection.
The original fully quantized backward remains an independently recorded control.
The decoded frozen-weight cache costs memory and is not a memory optimization.

An explicit follow-up option normalizes each token row by its FP32 maximum,
converts the normalized input to BF16, and quantizes with a fixed global maximum
of one, then rescales the BF16 GEMM result in FP32 and rounds it to BF16. This
unfused diagnostic removes cross-token scale dependence but is not the blog's
fused kernel or a claim of bit-exact TransformerEngine parity. Native decoded
arithmetic and an outlier-row independence canary must pass before model use.

Use the frozen Qwen3.5-4B revision, training inputs/Kimi targets, selected
320-example cohort and global longest-32 preflight from the B200 experiment.
Retain rank 128, alpha 256, AdamW 5e-5, logical batch 32, physical maximum 8,
16,384 padded-token budget, twelve checkpoints, SDPA, pinned FLA/convolution,
and the established selective compiler policy. Attention storage remains NF4
and compute BF16, as in the control. No generation KV cache or final test access.
Record actual hardware and cache state for every run. Match all comparison
conditions on the same GPU; do not transfer timings across GPU families.

Current target: original B200 Pod `alzfug70g5237b` in US-NC-2 at $6.79/hour,
with network volume `ixbh81vf9c`. Its preserved environment was verified after
resume: Python 3.12.3, Torch 2.11.0+cu130, GPU SM100, 183,359 MiB, zero volatile
uncorrectable ECC errors. Frozen inputs and kernel/model caches remain on the
volume; no data migration was required. Each campaign still verifies its own
input identities and native canaries. The infrastructure record documents the
[reservation handoff and unused stopped Pods](../../docs/infrastructure.md#runpod).

Stop on input drift, backend/kernel failure, OOM, nonfinite values, unchanged
adapters after updates, or failed numerical gates. Diagnostic loss comparisons
may record failures but never authorize training past them. No quality promotion
from short trajectories; separate grouped held-out validation remains required.
Inspect startup every 30–60 seconds. No in-chat scheduler is available, so
monitoring applies during the active turn only.

The separate `precision_cast_diagnostic.yaml` retains the original FP4 scaling
and backward while setting `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. The pinned
Torch source documents that default fusion can remove intermediate BF16
downcast/upcast pairs; the option preserves eager rounding boundaries. This is
a compiler hypothesis, not an established cause of the previous failure.
Ten-update campaigns automatically run a separate one-update global-longest-32
stage first for each precision condition, with zero warmup. Any failed stage
stops the campaign before the next stage.
`row_precision_cast_diagnostic.yaml` explicitly selects the matched 320 rows
with diagnostics only. Its ten-batch selection size never authorizes optimizer
updates or queues a global optimizer preflight. This checks input-dependent
compiler consistency after the first matched-cohort gate failure.

Initial implementation validation: 13 focused CPU tests passed using Torch
2.11.0+cpu and other exact locked dependencies in isolated `/tmp` overlays;
local CUDA/native dependency reads were stalling on Lustre. Ruff and shell syntax
checks passed. On the restored B200, original FP4 forward/backward canaries
passed at batch sizes 1/2/4/8. Per-token forward plus dequantized BF16 backward
also passed, with maximum relative L2 0.00235 forward and 0.00167 backward;
the outlier-row independence comparison had zero relative error. Artifacts are
`results/fp4_forward_diagnostic/kernel_canary.json` and
`results/fp4_native_stability/kernel_canary.json`. These are isolated arithmetic
checks; full-model numerical agreement and optimizer behavior are separate gates.

```bash
bash experiments/fp4_stability/launch.sh experiments/fp4_stability/config.yaml
```
