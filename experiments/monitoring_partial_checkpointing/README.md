# Monitoring partial checkpointing

## Question and frozen design

Can the compiled Qwen3.5-4B monitoring recipe avoid most linear-attention
activation recomputation while retaining safe headroom for the longest
trajectories? The control checkpoints all 24 linear-attention blocks and leaves
the eight full-attention blocks uncheckpointed. The candidate checkpoints only
linear blocks 2, 10, 18, and 26, leaving the other 20 linear blocks
uncheckpointed. Both retain `full_attention_and_linear_shell` compilation, with
the complete linear-attention mixer and its FLA and causal-conv1d kernels eager.

Before benchmarking, the candidate must process the fixed 32 longest examples
for one optimizer step and peak at no more than 72 GiB allocated on the 80 GiB
H100. This reserves over 7 GiB against transient and workload variation. Stop
on OOM, non-finite timing, adapter-key drift, missing fast-kernel binding,
compiler failure, checksum drift, more than 16 Dynamo graphs, or memory above
the ceiling.

After a passing preflight, the control and candidate run concurrently on the
two H100 SXM5 GPUs. They use the same checksummed 640-row monitoring selection,
sampling order, seed 0, rank-128 NF4 QLoRA, microbatch 1, accumulation 32,
standard AdamW, selected-position Kimi-soft BCE, FLA 0.5.2,
causal-conv1d 1.6.2.post1, and Triton 3.7.1. Each stops after eight optimizer
steps (256 examples). End-to-end Trainer examples/s, including cold compilation,
is primary. Select partial checkpointing only at a gain of at least 1%; reject
sub-1% changes. These systems-only adapters are not evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_partial_checkpointing.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_partial_checkpointing.run_lambda
```

Ignored artifacts are written under
`results/monitoring_partial_checkpointing/`.
