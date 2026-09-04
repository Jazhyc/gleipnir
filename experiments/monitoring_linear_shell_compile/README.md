# Monitoring linear-attention shell compilation

## Question and frozen design

Can Torch Inductor accelerate the non-kernel work in Qwen3.5's 24 checkpointed
linear-attention blocks without capturing FLA or causal-conv1d? The candidate
retains compilation of the eight uncheckpointed full-attention decoder
forwards, explicitly disables Dynamo at each linear-attention token-mixer
`forward`, and compiles the enclosing decoder-layer shell. This captures the
block input RMSNorm plus the post-mixer residual, RMSNorm, MLP/LoRA path, and
final residual. The mixer's projections, causal convolution, gated-delta scan,
gates, recurrent normalization, and output projection remain eager in this
first conservative screen.

The matched control uses full-attention-only compilation. Both conditions run
in parallel on separate H100 SXM5 GPUs with fresh compiler caches, the same
checksummed 640-row dataset and sampling order, seed 0, rank-128 NF4 QLoRA,
microbatch 1, accumulation 32, standard AdamW, selected-position Kimi-soft BCE,
FLA 0.5.2, causal-conv1d 1.6.2.post1, and Triton 3.7.1. Each condition stops
after eight optimizer steps (256 examples). End-to-end Trainer examples/s,
including cold compilation, is primary; paired step times are diagnostic.

Before the matched run, the candidate must complete one optimizer step over the
32 longest trajectories and pass the same-weights 2,048-token decision-logit
canary. Stop on checksum or recipe drift, OOM, backend failure, adapter-key
drift, a canary failure, missing fast-kernel binding, non-finite timing, more
than 16 Dynamo graphs, or step-count drift. Select the candidate at a gain of at
least 1%; sub-1% changes are rejected. These systems-only adapters are not
evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_linear_shell_compile.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_linear_shell_compile.run_lambda
```

Ignored artifacts are written under
`results/monitoring_linear_shell_compile/`.
