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
an eager-versus-compiled forward canary on the same model weights and 2,048-token
input keeps both decision logits within `0.05 + 0.01 * abs(logit)`, state
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

Version-2 artifacts are written under
`results/monitoring_selective_compile_v2/`.

## Superseded v1 preflight

The first preflight stopped because aggregate one-step Trainer loss differed by
0.00448 (0.84%) from a separate eager process. That comparison confounded
compiler numerics with ordinary cross-process mixed-precision variation: the
preceding checkpointing-only comparison itself changed reported loss by more
despite preserving the objective. Version 2 therefore replaces that indirect
gate with the same-process, same-weights decision-logit canary above. The v1
status and artifacts remain under `results/monitoring_selective_compile/`; v2
uses `results/monitoring_selective_compile_v2/`.

## Version-2 result

The same-weights canary passed: the maximum decision-logit difference was
0.00903 and the binary-margin difference was 0.00803. The matched 20-step run
reached 0.737 examples/s versus 0.676 for the eager selective-checkpointing
baseline and 0.673 for the original all-layer baseline. That is a 9.02%
incremental gain and a 9.51% combined gain. Mean steady optimizer-step time was
42.481 seconds, with one dynamic training graph and no shape-driven
recompilation. Peak allocated memory remained exactly 25.364 GiB. Adopt
full-attention selective compilation as the systems baseline.
