# Independent monitoring sequences in packed Qwen3.5 training

Status: CPU diagnostic passed; opt-in B200 BF16 LoRA integration under validation.
The preceding BF16 experiment has completed. Packing remains confined to this
bounded screen until the native isolation and numerical parity gates pass.

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
prompt rendering, truncation, seed, FP32 rank-128 adapters, frozen BF16 base, optimizer,
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
calls, peak memory, compile count and setup costs separately. The CPU entrypoint
does not launch or schedule this stage.

See the [source audit and integration design](../../docs/research/monitoring_sequence_packing.md)
for model/kernel changes and the prior Phoenix rejection.

## Bounded BF16 B200 entrypoint

```bash
bash experiments/fp4_stability/launch.sh experiments/monitoring_sequence_packing/bf16_gpu.yaml
```

This reuses the completed BF16 LoRA recipe, its pinned FlashQLA precision
boundary, original initial adapter and hashed 320-example cohort. The screen
installs segmented causal SDPA for full attention, passes `seq_idx` to native
convolution and cumulative lengths to FlashQLA, and resets positions per example.
The same SDPA router is installed for both controls. No dense long-context mask
is constructed. Packing is best-fit within each logical update of 32 examples,
with a 16,384-token budget and intact oversized singletons.

Before either trajectory, and again after compilation, check identical-shape
prefix perturbations (maximum absolute decision-logit drift 1e-6), checkpointed
cross-example input gradients (maximum 1e-8, nonzero own-example effect), and
packed versus singleton monitoring losses (absolute 0.02 plus relative 0.02)
and adapter gradients (relative L2 0.05). Any failure stops before optimizer
updates. Neither isolation nor speed alone establishes held-out quality parity.
Each condition includes longest-example preflight, ten no-update warmup batches
and ten measured updates with fresh AdamW state. Report setup separately from
measured steps, preserve FP32 masters and compare all ten batches. Artifacts go
under `results/bf16_sequence_packing/`; logs go under
`logs/runpod/bf16_sequence_packing/`.

The first B200 attempt reached the numerical gate after all three isolation
cases passed. Its packed/singleton losses were 0.939848/0.957634 and adapter
gradient relative L2 was 0.0581823, exceeding 0.05; it stopped with zero updates.
`bf16_stable_gpu.yaml` tests whether disabling PyTorch's intermediate BF16 GEMM
reductions removes this shape-dependent numerical drift. This intervention is
applied to both controls, with the same BF16 storage/activations and FP32 masters.
The gate and its tolerances remain unchanged. Receipts now retain layerwise
activation differences and all successful isolation cases even on later failure.

Disabling reduced-precision GEMM reductions alone reproduced the original values
exactly. Its retained receipts show zero cross-example influence/gradients and
an exact first-layer recurrent-mixer match. The first discrepancy appears in
the first MLP/residual output (maximum 0.00048828125), then grows through depth.
`bf16_no_splitk_gpu.yaml` additionally disables split-K GEMM reductions: the
installed Torch API's boolean `False` preserves split-K, whereas `(False, False)`
disables both controls. This keeps the same BF16 parameters and activations.
Both controls use identical settings, and all original gates remain fixed.
The packing launcher skips the unrelated FP4 arithmetic preflight because this
screen has no FP4 modules; native convolution/recurrence checks still run on
the real BF16 model before training. This changes setup, outside measured steps.

Torch requires the cuBLASLt backend when split-K is disabled; the first no-split-K
attempt failed at that runtime requirement with zero updates. The BF16 control
helper now explicitly selects and records cuBLASLt for this setting.
`bf16_no_splitk_lt_gpu.yaml` runs that corrected common backend configuration.

That corrected backend passed the eager model canary: every recorded layer
output and the loss matched exactly, with adapter gradient relative L2 0.00888195
and zero cross-example dependence. The compiled run then stopped on a diagnostic
hook collection error with zero updates. Layer hooks are now confined to eager
execution; all compiled loss/readout/input-gradient/adapter-gradient gates remain
unchanged. `bf16_checked_gpu.yaml` repeats the corrected complete protocol.

The corrected compiled check retained zero sequence leakage but failed numerical
parity (gradient relative L2 0.0880173; singleton/packed losses
0.949228942/0.957633674). No updates ran. `bf16_casts_gpu.yaml` additionally
preserves intermediate precision casts in Inductor for both conditions, while
retaining the same compile/checkpoint policies and fixed gates.

Preserving casts also failed compiled numerical parity (gradient relative L2
0.143217; singleton/packed losses 0.949228942/0.967125356), with no leakage and
zero updates. `bf16_eager_gpu.yaml` therefore measures the validated native eager
implementation against a matched eager padding control. Torch model compilation
is disabled in both; native convolution/FlashQLA kernels remain enabled and
checkpointing, data, adapter, optimizer and all gates remain unchanged. This
condition does not promote or imply acceptance of compiled packing.
