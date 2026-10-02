# Independent monitoring sequences in packed Qwen3.5 training

Status: CPU and native B200 isolation checks passed, with completed eager and
compiled BF16 LoRA comparisons. New comparisons use packing only. Longer
training and held-out quality validation remain pending.

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

## Historical paired BF16 B200 entrypoint

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

## Compiled learning diagnostic

The user authorized a bounded training check despite the 14.3% gradient
disagreement. `bf16_learning_gpu.yaml` repeats the cast-emulation compiled recipe
with an explicit `packing_learning_gradient_tolerance: 0.15`. The existing
strict 0.05 parity result remains recorded as failed; a separate
`accepted_for_learning_comparison` field permits the diagnostic. No isolation,
finite-gradient, or loss-agreement threshold changes. Gradients above 0.15,
nonfinite values, or cross-example influence still stop before updates.

Hypothesis: shape-dependent BF16 gradient drift can remain stable over a short
training trajectory. Compare the same initial FP32 adapters and ten logical
batches with fresh AdamW state for adaptive padding and packing. Examine finite
losses/gradient norms and the common original-backend probe after each update,
alongside complete-loop timing and memory. Stop on any nonfinite loss/gradient
or failed preflight gate. Ten updates diagnose early stability; they do not
establish convergence or held-out quality equivalence. The first scheduled
step has zero learning rate, leaving nine weight-changing updates per control.
Both common probes use the original padded collator and original kernels/forwards
so comparison does not depend on the training layout.

The eager comparison completed: 9.3926 seconds/update padded versus 7.2430 packed
(22.89% lower), 115 versus 74 physical calls and effectively identical 118.42 GiB
peak allocated memory. Every loss and gradient norm remained finite. Its probes
used each training collator and had slightly different starting losses, so their
final values (0.79265 padded, 0.76925 packed) are rough stability diagnostics.
See the [finding](../../docs/findings/bf16_sequence_packing.md) for scope and limits.

The initial compiled learning attempt reproduced the 14.3217% gradient
disagreement and accepted it for the diagnostic. During longest-example
preflight, Torch exhausted `recompile_limit=8` for the linear decoder shell
(different `kwargs` counts after boundary canaries) and fell back to eager.
The attempt was interrupted with zero optimizer updates; its receipts remain
under `results/bf16_sequence_packing_learning/` with an interruption record.
`bf16_learning_cached_gpu.yaml` retains the same recipe and explicitly sets
`packing_compile_cache_limit: 64` in both controls. It also enables
`fail_on_recompile_limit_hit` so cache exhaustion fails instead of silently
changing execution. Both settings are recorded and restored after the screen.

That corrected compiled comparison completed both trajectories with no fallback
warnings and no new Dynamo graphs during measured steps. Its packing canaries
passed the strict 5% gate (gradient relative L2 0.00885904); both compiled losses
were 0.957633674. Mean updates were 7.7928 seconds padded and 5.7163 packed
(26.65% reduction), with about 98.04 GiB peak allocated in both. Shared probes
started identically at 1.20367 and ended at 0.77854/0.77345; every loss/gradient
norm remained finite. The improvement over the earlier gradient receipt is
observed, but cache-limit causality is not established by these two runs.

## No-checkpoint follow-up

After the compiled comparison, the user requested testing removal of gradient
checkpointing to use B200 memory and reduce recomputation. Hypothesis: the
unchanged physical batches fit without checkpointing and complete faster.
`bf16_no_checkpoint_gpu.yaml` repeats both compiled padded/packed controls with
checkpointing disabled, preserving the same initial adapter, cohort, order,
optimizer, precision boundary and compiler cache controls. The source job records
`gradient_checkpointing=false`, policy `all` (inactive) and no layer selection;
the model receipt must have an empty checkpointed-layer list.

Run the existing longest-32 preflight first, including the 28,733-token maximum
in this cohort. Stop on OOM, nonfinite values or any isolation/parity failure
beyond the declared diagnostic policy; do not shrink batches to hide an OOM.
Compare complete update times and peak allocated memory against the checkpointed
compiled controls. Require at least 5% lower update time before recommending the
memory intervention. No held-out selection changes or quality promotion occur.

The follow-up passed with zero checkpointed layers, zero measured sequence
leakage and compiled canary gradient relative L2 0.00697903. Both longest-example
preflights and all ten-step trajectories completed without nonfinite values or
compiler fallback. Mean updates were 7.1180 seconds padded and 5.1452 packed,
8.66%/9.99% below their checkpointed controls. Peak allocated memory rose from
98.04 to 147.14 GiB and fit the existing B200. The initial hashes, logical batches,
physical partitions, learning rates and shared-probe starting values matched.
This supports the no-checkpoint configuration as an opt-in systems candidate;
longer convergence and frozen held-out quality checks remain necessary.

## Larger physical batches with checkpointing

The user proposed spending memory on larger batches instead of disabling
checkpointing. Hypothesis: fewer physical calls with checkpointing retained
outweigh recomputation cost. `bf16_larger_batch_gpu.yaml` retains the original
12 checkpointed layers and doubles the physical token budget from 16,384 to
32,768, using packing only as requested. Packed rows are already
single batch rows with variable example counts; their size is controlled by
the token budget. The logical batch remains 32 with unchanged initial adapters,
cohort/order, targets, learning rate and optimizer schedule.

Run longest-32 preflight and the same ten warmup/ten measured updates with
packing. Stop on OOM, nonfinite values, leakage or the recorded
packing gates, without shrinking the declared batch. Compare update time,
physical calls and peak allocated memory against the existing packed
checkpointed 16,384-token run and the packed no-checkpoint run. Require at
least 5% lower packed update time than 5.1452 seconds (the no-checkpoint result)
before favoring the larger checkpointed batch. Preserve all negative results;
this is a systems comparison and makes no held-out quality claim.

The packed-only 32k run completed with 44 physical calls versus 74 at 16k.
Mean update time was 5.5162 seconds at 111.04 GiB peak allocated memory: 3.50%
faster than checkpointed 16k packing, but 7.21% slower than 16k packing without
checkpointing (5.1452 seconds, 147.14 GiB). It failed the prospective 5% speed
gate against no checkpointing. Isolation and strict compiled numerical gates
passed, all values stayed finite, and no measured recompilation/fallback occurred.
The shared probe ended at 0.77203 from the same 1.20367 start. This budget did
not make checkpointing faster than removing recomputation; other budgets remain
untested.

New comparisons default to `packing_only: true`; the launcher forwards this
explicitly and receipts record it. Historical paired configurations retain
`packing_only: false` for reproducibility. Independent singleton numerical and
sequence-isolation canaries remain mandatory, as does the shared bounded probe.

For future comparisons use the packing-only configuration above (with a new
output/log destination). Paired configurations in this README document historical
protocols and should only be used for an explicitly requested replay.
