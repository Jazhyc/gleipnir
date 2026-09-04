# Monitoring gradient-checkpointing screen

## Question and frozen design

Does disabling activation gradient checkpointing make the selected
Qwen3.5-4B monitoring-only recipe materially faster while still fitting the
longest trajectories on one 80 GiB H100? The matched baseline is the completed
`mb1-random` condition from `monitoring_training_throughput`: rank-128 NF4
QLoRA, microbatch 1, accumulation 32, selected-position Kimi-soft BCE, pinned
FLA 0.5.2 and causal-conv1d 1.6.2.post1, with eager execution.

The intervention changes only gradient checkpointing. It first processes the
32 longest monitoring trajectories for one effective-batch optimizer step. If
that preflight fits, the throughput condition processes the exact same
checksummed 640 rows and 20 optimizer steps as the baseline. End-to-end
examples/s is primary; unpadded tokens/s, synchronized step time, and peak CUDA
memory are diagnostics. Adopt checkpointing-off only if it is complete and at
least 5% faster than the completed checkpointed baseline.

Stop on checksum drift, a non-monitoring row, missing fast-kernel binding, OOM
in the longest-trajectory preflight, non-finite loss or timing, anything other
than the declared optimizer-step count, or effective-batch drift. These are
systems-only adapters and are not evaluated or promoted.

PyTorch SDPA is already the automatically selected backend for Qwen3.5's eight
full-attention layers. FlashAttention-2 is not installed on the Stack 24 node;
three quarters of the layers use the pinned FLA recurrent path, so a separate
attention-package installation is deferred unless checkpointing leaves a
material unresolved bottleneck. True trajectory packing remains deferred until
full-attention masks and gated-delta/convolution states reset at every example
boundary.

## Run

```bash
python -m experiments.monitoring_gradient_checkpointing.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_gradient_checkpointing.run_lambda
```

Ignored artifacts are written under
`results/monitoring_gradient_checkpointing/` and runtime logs under
`logs/lambda/monitoring_gradient_checkpointing/`.

## Result

The Stack 24 H100 longest-trajectory preflight failed closed on its first
microstep. With checkpointing disabled, PyTorch had allocated 77.85 GiB and the
process occupied 78.91 of 79.18 GiB when a further 282 MiB allocation failed.
No 20-step throughput condition or adapter was produced. The completed baseline
already resolves Transformers checkpointing to its recommended non-reentrant
implementation, so there is no lower-risk checkpointing-mode switch left to
screen. Retain gradient checkpointing for the full-length recipe.
