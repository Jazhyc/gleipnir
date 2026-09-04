# Monitoring-training throughput screen

## Question and frozen design

Can the Qwen3.5-4B monitoring-only recipe process the same long-trajectory
examples faster by using microbatch 1 or length-aware microbatch 2? The stopped
learning-rate sweep showed about 92 seconds per effective-batch-32 optimizer
step with random microbatch 2, slower than the historical microbatch-1 run.

The three conditions are `mb1-random` (microbatch 1, accumulation 32),
`mb2-random` (the stopped-sweep baseline), and `mb2-length-grouped`
(microbatch 2, accumulation 16). Each uses seed 0, rank-128 QLoRA, AdamW at
`5e-5`, selected-position Kimi-soft BCE, pinned FLA 0.5.2, causal-conv1d
1.6.2.post1, and isolated Triton 3.7.1. Each processes the same checksumed 640
monitoring examples for exactly 20 optimizer steps, so the effective batch and
example exposure remain fixed.

The trainer records end-to-end samples/s, synchronized per-optimizer-step wall
times, peak memory, and both padded and unpadded token counts. The primary
systems metric is end-to-end examples/s; unpadded direct tokens/s, padding
fraction, and warmup-excluded median step time diagnose the result. Replace the
random microbatch-2 baseline only for a complete condition that is at least 5%
faster. Ties within 2% prefer the lower-padding recipe and then microbatch 1.

True sequence packing is out of scope for this screen. Qwen3.5 interleaves full
attention with recurrent gated-delta and causal-convolution layers; safe packing
requires explicit attention and recurrent-state resets at every trajectory
boundary plus multi-boundary selected-logit extraction. The current collator
does not provide those semantics, so concatenating examples would leak state.

Stop on checksum drift, a non-monitoring row, missing kernel binding, OOM,
non-finite loss/timing, anything other than 20 completed steps, or effective
batch drift. These are systems-only adapters and are not evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_training_throughput.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_training_throughput.run_lambda
```

Ignored artifacts are written under
`results/monitoring_training_throughput/` and runtime logs under
`logs/lambda/monitoring_training_throughput/`.
