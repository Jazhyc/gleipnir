# Monitoring inner-mixer compilation

## Question and frozen design

Can Torch Inductor accelerate the non-kernel segments inside Qwen3.5's 24
linear-attention mixers? The current control compiles all decoder-layer shells
but explicitly leaves each complete linear mixer eager. The candidate instead
places graph breaks around `causal_conv1d_fn`, `chunk_gated_delta_rule`, and
FLA's fused gated-normalization forward. Inductor may therefore capture the
input QKV/Z/A/B projections and LoRA paths, reshapes, gates, repeat operations,
output projection, decoder norms, MLP/LoRA, and residuals. All pinned
causal-conv1d and FLA custom kernels themselves remain eager.

Both conditions checkpoint all 24 linear-attention blocks and leave all eight
full-attention blocks uncheckpointed. They run concurrently on separate H100
SXM5 GPUs with fresh compiler caches, the same checksummed 640-row selection
and order, seed 0, rank-128 NF4 QLoRA, microbatch 1, accumulation 32, standard
AdamW, selected-position Kimi-soft BCE, FLA 0.5.2,
causal-conv1d 1.6.2.post1, and Triton 3.7.1. Each condition stops after eight
optimizer steps (256 examples). End-to-end Trainer examples/s including cold
compilation is primary; warmup-excluded step timing is diagnostic.

Before the comparison, the candidate must complete one optimizer step over the
fixed 32 longest trajectories and pass the same-weights 2,048-token
decision-logit canary. Stop on checksum or recipe drift, OOM, a backend or
kernel failure, adapter-key drift, canary failure, non-finite timing, more than
24 Dynamo graphs, or step-count drift. Select the candidate only at a gain of
at least 1%; sub-1% changes are rejected. These systems-only adapters are not
evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_inner_mixer_compile.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_inner_mixer_compile.run_lambda
```

Ignored artifacts are written under
`results/monitoring_inner_mixer_compile/`.
