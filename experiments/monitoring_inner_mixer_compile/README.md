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

## Result

The first preflight attempt also compiled FLA's fused gated-normalization
forward. TorchDynamo entered the custom autograd implementation, encountered
an unknown CUDA device-properties builtin, and warned of potential silent
incorrectness. That attempt was stopped before benchmarking and its Lambda
artifacts were retained under the `_unsafe_norm_boundary` suffix. The frozen
candidate therefore keeps the gated norm eager along with causal-conv1d and
the gated-delta scan.

The corrected preflight passed at commit `8495c28`: the maximum absolute
decision-logit difference was 0.02165, the binary-margin difference was
0.00131, and all three custom-kernel boundaries were recorded. In the matched
eight-step screen, the shell control reached 0.815 examples/s and the inner
mixer candidate reached 0.798 examples/s, a 2.09% end-to-end regression. The
candidate's warmup-excluded mean step time was effectively tied and slightly
better (36.87 versus 36.96 seconds, 0.23%), but it required seven Dynamo graphs
instead of three and paid more cold compilation. Peak allocation was unchanged
at 24.582 GiB. Reject the candidate and retain
`full_attention_and_linear_shell`.
