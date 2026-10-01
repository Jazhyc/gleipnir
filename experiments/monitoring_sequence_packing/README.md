# Independent monitoring sequences in packed Qwen3.5 training

Status: CPU diagnostic and integration design only. Do not start B200 work until
the current FlashQLA/FLA experiment is complete. This experiment does not change
the ordinary training recipe or enable sequence packing in its runner.

## Hypothesis and contract

Flattening several monitoring prompts into one physical row can remove padding
and reduce model calls without allowing any example to condition on another.
This requires attention isolation, independent convolution context and independent
gated-delta state in both forward and backward. Changing preceding example A must
leave every hidden state and decision score for B unchanged within a frozen
numerical tolerance; a loss computed only on B must have zero input gradients
on A. Shared model-parameter gradients are expected and are not a leakage test.

The intervention is an opt-in boundary-aware collator and packed readout. Keep
the same logical updates of 32 equally weighted examples, source/teacher hashes,
prompt rendering, truncation, seed, FP32 rank-128 adapters, NF4/BF16, optimizer,
learning-rate schedule and checkpoint policy. Never divide the monitoring loss
by token count or number of packed rows. Keep oversized examples intact as
singletons. Sorting/packing is confined to the current logical update.

Baselines are independent singleton execution and current 16,384-token/max-eight
adaptive padded batching, crossed with the same attention/recurrent backend.
Do not confound a kernel replacement with packing. Correctness canaries use
training inputs and synthetic examples, not held-out selection. Any later quality
campaign must freeze the agreed CoT-removed ID validation contract; final test
results cannot promote this systems intervention.

Stop before timing on nonfinite values, missing boundaries, cache/state reuse,
unsupported/fallback kernels, failed independence, decision readout mismatch,
source drift or OOM. Freeze B200 tolerances against identical-shape repeats before
testing candidates; preserve the historical 0.05 gradient gate and record any
separately authorized learning diagnostic explicitly. No GPU parity or training
quality equivalence follows from the CPU reference.

## CPU entrypoint

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m experiments.monitoring_sequence_packing.audit_cpu
.venv/bin/pytest tests/test_packed_sequences.py
```

`config.yaml` selects a deterministic, random-weight, float32 two-layer CPU
Qwen3.5 with one Gated DeltaNet layer and one full-attention layer. No model
checkpoint, dataset download, GPU, optimizer or remote execution is used. The
artifact records the installed model source hash, versions, tolerances, output
differences and cross-example input gradients under
`results/monitoring_sequence_packing/cpu_isolation.json`.

The diagnostic supplies all boundaries while deliberately leaving both Torch
fallbacks unsegmented, then enables each reset separately. A separate negative
control enables both resets but supplies an all-ones 2D attention mask. Finally,
two positive controls enable both resets with either automatic packed-attention
detection or an explicit boolean block-diagonal causal mask. The reset wrappers
split CPU reference operators at boundaries and restore their model bindings on
exit; they are correctness oracles, not candidate high-throughput kernels.

The seed-0 `[7,5]` CPU probe completed: both fully isolated controls exactly
matched standalone B and had zero A-to-B output dependence and input gradients;
all four deliberately incomplete configurations showed leakage. These are
random-weight CPU reference results, not B200 kernel parity or quality evidence.

The bounded explicit mask and layout builder live in
`src/gleipnir/packed_sequences.py`. The default dense-mask limit is 2,048 tokens;
never build a quadratic dense mask over a production 16k/30k packed row.

## Next B200 stage

After the current experiment completes, start with isolated native convolution,
FLA and FlashQLA forward/backward probes using sequence lengths around convolution
width four and scan/chunk boundaries 63/64/65 and 127/128/129. Include length-one
sequences, unequal lengths, reordered packs and perturbations of preceding
examples. Record hashes/versions of the installed native kernels. Start with
FlashQLA automatic intra-card splitting disabled, then test it independently.

Next run a short frozen-model/adapter canary, collecting intermediate outputs
at every convolution, recurrent mixer and full-attention layer. Find the first
diverging layer before patching any kernel. Check packed-versus-independent
decision logits, probabilities, per-example losses, input isolation and adapter
gradients. Repeat with checkpointing and selective compilation because backward
recomputation must receive identical boundaries. Verify a nonzero adapter effect.

Only after those gates pass, compare complete warmed training loops on the
existing stratified 320-example mixed cohort. Require at least 5% lower complete
loop time over a matched backend/control; report real/padded tokens, physical
calls, peak memory, compile count and setup costs separately. This stage is not
implemented or scheduled by the CPU entrypoint.

See the [source audit and integration design](../../docs/research/monitoring_sequence_packing.md)
for model/kernel changes and the prior Phoenix rejection.
