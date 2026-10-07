# Training instructions

These instructions are required for the tasks routed here by the root
[AGENTS.md](../../AGENTS.md). Standing preferences were moved from that
file on 2026-10-07; their scope and historical acceptance remain intact.

Also read [compute.md](compute.md) before launching or operating a run.

## Reuse validated training recipes

Standing user preference, 2026-10-06: short systems-training optimization checks
overwrite a shared scratch adapter/master instead of retaining weights for
every condition. Keep per-trial configurations, timings, losses, gradient
diagnostics, tensor digests and numerical receipts. Preserve one frozen original
initialization for matched resets, compiler/kernel caches, and full research
training/evaluation adapters. Expendable historical systems weight copies were
explicitly authorized for deletion; this does not authorize deleting quality
artifacts or caches. Use `results/systems_training_scratch/` for mutable weights
and clearly mark receipts that point there as references to an overwritten slot.
For ordinary Trainer screens, set `systems_adapter_scratch: true` in the screen
configuration/job; this suppresses checkpoint copies and redirects only weights,
while metadata stays per trial. Keep historical frozen configs unchanged.

Standing user preference, 2026-10-05: keep a persistent, resident training worker
when iterating on training-stack optimizations. Reuse the loaded model, compiler
and kernel caches, native plans and packed frozen weights across compatible
trials. Reset FP32 adapters, optimizer/scheduler state, RNG and data order for
matched comparisons; verify reset correctness when establishing a worker.
Do not reload the model, repeat full-shape replay or rerun unchanged startup
diagnostics for every trial. Keep finite/missing-gradient update checks and run
targeted checks when an intervention changes arithmetic or the supported envelope.
Record the worker PID, active configuration and artifact/queue paths so later
sessions can inspect and reuse it. Restart when changes require it or after a
diagnosed failure; preserve artifacts and disk caches. This preference does not
authorize launching new billable capacity or terminating existing capacity.
Use the combined native FP4 MLP plus BF16 GDN/FA4 variant as the optimization
baseline and, following the user's later explicit selection on 2026-10-05,
the default B200 training recipe. This selection does not establish held-out
quality equivalence with BF16.

Standing user preference, 2026-10-02: once a recipe has been validated, reuse
its recorded startup validation and begin ordinary training without repeating
numerical comparison canaries, packing isolation probes or longest-batch memory
preflights. This instruction supersedes the repeated-startup-gate requirements
below for unchanged validated recipes. Supported hard/soft binary loss mixtures
and learning-rate sweeps do not by themselves require repeating those probes.
Rerun diagnostics when hardware, kernel/compiler versions, precision, batching,
packing code, supported objective family or context/memory envelope materially
changes, when failures give reason to doubt the recipe, or when requested.
Keep input/provenance checks and finite/missing-gradient checks during updates.
Record the validation reference and its checksum; mark reused/skipped checks
explicitly instead of claiming a new pass. Preserve historical failed receipts.
Serving artifacts still need adapter-specific score parity before evaluation.

## H100 Qwen3.5 training

For text-only Qwen3.5 training on H100s, do not silently use Transformers'
Torch gated-delta fallback. Use the causal-LM loader and verify the isolated,
pinned `flash-linear-attention==0.5.2` and `fla-core==0.5.2` kernels before model
import. The proven eager recipe is microbatch 8 with gradient accumulation 4
(effective batch 32) and `torch.compile=false`; preserve that recipe unless a
matched benchmark supports a change. Prefer standard QLoRA (4-bit NF4, double
quantization, BF16 compute) when it permits the proven batch at high adapter
ranks. Preflight the largest rank, fail closed if FLA is unavailable, and record
kernel, quantization, batch, memory, and throughput metadata for every campaign.
For direct-boundary objectives, avoid materializing full-sequence vocabulary
logits when it prevents the proven batch; use a matched, recorded selected-token
projection consistently across the campaign.

## Single-B200 Qwen3.5-4B training

For single-B200 Qwen3.5-4B training, use the user-selected
`systems_screen@_global_: qwen35_4b_b200_default` profile: all 24 GDN layers
use pinned FlashQLA with BF16 Q/K/V and FP32 gates/normalization; full attention
uses pinned FlashAttention 4 4.0.0b33 with native causal variable-length packing.
Load an unquantized BF16 base, use no model checkpointing, and keep the
16,384-token packing budget, logical batch 32, and FP32 master adapters.
Use native NVFP4 MLPs with hardware activation packing and fused descale in
forward and base input gradients. Keep GDN projections/recurrence and full
attention in BF16. Reuse the checksum-bound warmed03 FP4 receipt for compatible
runs with explicit `selected_finite` acceptance; preserve its failed loss and
strict 5% gradient comparisons and passing isolation/preflight separately.
Reject missing/nonfinite gradients before updates. Keep the historical BF16 FA4
recipe as `qwen35_4b_b200_bf16_fa4`, including its separate accepted 10% ceiling,
and preserve original-FLA comparison profiles. Do not change frozen campaigns'
recorded precision/receipts. See `docs/decisions/b200_native_fp4_training_recipe.md`.

Read the [native FP4 recipe decision](../decisions/b200_native_fp4_training_recipe.md)
before preparing a B200 run. It binds the startup receipt, source/runtime
checks and accepted numerical limits.
