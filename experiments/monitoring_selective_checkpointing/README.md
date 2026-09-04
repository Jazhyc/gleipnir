# Monitoring selective-checkpointing screen

## Question and frozen design

Can unused H100 memory reduce long-context backward recomputation without the
OOM caused by disabling checkpointing entirely? Qwen3.5-4B has 24
linear-attention decoder layers and eight full-attention layers. The candidate
checkpoints the 24 linear-attention layers but leaves every fourth,
full-attention layer uncheckpointed. The intervention preserves model inputs,
outputs, loss, optimizer, and update order.

The baseline is the completed direct-only `adamw-torch` condition from
`monitoring_adamw_throughput`. The candidate uses the same checksummed 640
examples for 20 optimizer steps, seed 0, rank-128 NF4 QLoRA, microbatch 1,
accumulation 32, standard AdamW, eager execution, selected-position Kimi-soft
BCE, FLA 0.5.2, causal-conv1d 1.6.2.post1, and Triton 3.7.1. It must first
complete one optimizer step over the 32 longest trajectories.

Select the candidate only if it completes, remains below the 80 GiB limit, and
is at least 5% faster in end-to-end examples/s. Stop on checksum or recipe
drift, OOM, missing kernel binding, non-finite loss/timing, or step-count drift.
These systems-only adapters are not evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_selective_checkpointing.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_selective_checkpointing.run_lambda
```

Ignored artifacts are written under
`results/monitoring_selective_checkpointing/`.

## Result

The longest-32 preflight completed, and the matched 20-step condition reached
0.676 examples/s versus 0.673 for all-layer checkpointing: a 0.45% end-to-end
gain below the 5% threshold. Excluding two warmup steps, mean optimizer-step
time moved from 47.946 to 47.706 seconds, a similarly negligible 0.50% change.
Both conditions peaked at exactly 25.364 GiB allocated and 28.424 GiB reserved.
Retain all-layer checkpointing; the eight full-attention layers are not a
material checkpoint-recomputation bottleneck under this recipe.
