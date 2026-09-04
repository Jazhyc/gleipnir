# Monitoring AdamW throughput screen

## Question and frozen design

Can PyTorch's fused AdamW improve the selected Qwen3.5-4B monitoring-training
recipe over the standard PyTorch AdamW implementation? The two conditions use
the exact same checksummed 640 examples for 20 optimizer steps on separate H100
SXM5 GPUs. Both hold seed 0, rank-128 NF4 QLoRA, microbatch 1, accumulation 32,
gradient checkpointing, eager execution, selected-position Kimi-soft BCE, FLA
0.5.2, causal-conv1d 1.6.2.post1, and Triton 3.7.1 fixed. The only intervention
is `adamw_torch` versus `adamw_torch_fused`.

This screen also uses the semantics-preserving direct-only input materializer:
because completion loss is zero, it does not tokenize, pad, or transfer unused
completion IDs and labels. The direct prompt token IDs and objective are
unchanged in both arms. This primarily targets per-cell preparation and host
overhead rather than GPU model compute.

End-to-end examples/s is primary; synchronized optimizer-step time, unpadded
tokens/s, peak memory, and total wall time are diagnostics. Select fused AdamW
only if it completes and is at least 3% faster. Before the matched run, fused
AdamW must complete one optimizer step over the 32 longest trajectories.

Stop on checksum drift, missing kernel binding, OOM, non-finite loss or timing,
anything other than the declared step count, effective-batch drift, or
materialization of completion tensors. These systems-only adapters are not
evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_adamw_throughput.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_adamw_throughput.run_lambda
```

Ignored artifacts are written under `results/monitoring_adamw_throughput/` and
runtime logs under `logs/lambda/monitoring_adamw_throughput/`.

## Result

Both conditions completed. Standard AdamW reached 0.673 examples/s and fused
AdamW reached 0.676 examples/s on identical exposure, a 0.45% difference below
the 3% adoption threshold. Their peak allocations were both 25.36 GiB and their
steady mean step times were 47.95 and 47.55 seconds respectively. Retain
`adamw_torch`; long-context model compute, not the once-per-32-microstep
optimizer update, dominates this recipe.

The direct-only materializer's standard arm was about 0.9% faster than the
earlier 0.667 examples/s microbatch-1 condition. Treat the GPU-rate difference
as marginal; retain the cleanup because it removes an entire unused completion
tokenization, tensor materialization, padding, and host-to-device transfer from
every soft-direct-only job without changing direct token IDs or the loss.
