# Monitoring selective-compilation screen

## Question and frozen design

After selective activation checkpointing is measured, can compilation remove
additional per-token overhead without trying to capture the proven FLA and
causal-conv kernels? The candidate compiles the `forward` method of only the
eight uncheckpointed full-attention decoder layers with Torch Inductor,
`dynamic=true`, and `fullgraph=false`. Global compilation remains disabled.

The baseline is the completed `linear_attention_only` checkpointing condition.
The candidate uses the same checksummed 640 examples for 20 optimizer steps,
seed 0, rank-128 NF4 QLoRA, microbatch 1, accumulation 32, standard AdamW,
selected-position Kimi-soft BCE, FLA 0.5.2, causal-conv1d 1.6.2.post1, and
Triton 3.7.1. It first trains one optimizer step over the same 32 longest
trajectories used by the eager selective-checkpointing preflight.

Run the candidate once the eager selective-checkpointing benchmark completes,
even if that component alone misses its selection threshold, because compilation
may interact with the newly uncheckpointed regions. Select the combined recipe
only if compilation adds at least 5% over its eager selective-checkpointing
baseline, the combination is at least 5% faster than the original all-layer
checkpointing baseline, the longest-trajectory preflight completes below 80 GiB,
its loss agrees with the eager preflight within
`5e-4 + 1e-3 * abs(eager_loss)`, state
dictionary keys remain unchanged, Dynamo reports no more than 16 unique graphs,
and end-to-end examples/s improves by at least 5%. Stop on checksum or recipe
drift, OOM, backend compilation failure, missing kernel binding, non-finite
loss/timing, excessive graph specialization, or step-count drift. These
systems-only adapters are not evaluated or promoted.

## Run

```bash
python -m experiments.monitoring_selective_compile.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_selective_compile.run_lambda
```

Ignored artifacts are written under `results/monitoring_selective_compile/`.
