# Packed BF16 LoRA as the B200 training default

Status: selected by explicit user instruction, 2026-10-02.

Kernel screen, 2026-10-04: [FA4's native canary passed, but eager model packing
parity failed](../findings/b200_bf16_fa4.md) at 8.29% adapter-gradient relative
L2 against the unchanged 5% gate. No FA4 optimizer updates or valid timing
comparison followed from that attempt. After inspecting zero leakage and the
layer-level differences, the user authorized a separate timing continuation with
a 10% learning acceptance ceiling, retaining strict failures. Keep segmented
SDPA as the default pending replication and quality validation. The authorized
20-update continuation observes 20.24% shorter measured update time with FA4
(5.12363 to 4.08648 seconds), with identical physical partitions/tokens. Fresh
FA4 checks front-load compilation and make its complete invocation longer.

Startup validation update, 2026-10-02: the user explicitly requested reusing
validation for an unchanged recipe instead of repeating diagnostic gates on
every run. Reuse the recorded kernel/packing/partition/memory receipts and mark
checks as not performed in the new run, with reference identity and checksum.
The fresh-validation procedure below applies when a recipe or its hardware,
kernels, precision, batching, packing or supported context envelope materially
changes, after relevant failures, or on request. Supported binary loss mixtures
and LR sweeps can reuse validation. Keep finite/missing-gradient checks during
updates and adapter-specific serving parity. See `AGENTS.md` and
`experiments/monitoring_hard_labels/README.md`.

Use `systems_screen@_global_: qwen35_4b_b200_default` for new single-B200
Qwen3.5-4B binary monitoring training. The user selected the fastest completed
packed recipe after reviewing the checkpointing/batch-size comparison. This
updates the execution default without launching a full training campaign or
changing its data, targets, duration or held-out checkpoint-selection rule.

The profile selects a frozen, unquantized BF16 causal-LM base and FP32
rank-128/alpha-256 master adapters, with model gradient checkpointing disabled.
Keep logical batches of 32 examples, accumulation one, and best-fit packing
within each logical batch at 16,384 input tokens per physical row. Oversized
examples remain intact singletons under the existing 29,696-token context cap.
Losses retain equal example weighting, including a final incomplete batch.
Retain AdamW 5e-5, norm-one clipping, linear scheduling and 3% warmup. Campaigns
still explicitly choose training duration and input/teacher identities.

Every convolution receives `seq_idx`; every FlashQLA scan receives cumulative
sequence lengths; positions reset per example. Full attention uses independent
causal SDPA calls per segment without a quadratic long-context mask. All 24
linear layers use pinned FlashQLA revision
`da06429d54b0f577de0a638f451ac8f0b395e0ac`, automatic partitioning disabled,
BF16 Q/K/V and FP32 gates/QK normalization. Preserve its kernel checkpoint cache.
FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1 and Triton 3.7.1 remain required.

Keep the tested selective Inductor policy `full_attention_and_linear_shell`,
precision-cast emulation, cuBLASLt and disabled BF16 reduced-precision/split-K
reductions. Use compiler recompile limit 64 and fail on exhaustion. Runtime
settings and the attention router are scoped and restored on success/failure.
The cache limit controls dispatch; it is not an accuracy setting.

Before optimizer updates, verify BF16 bases/FP32 masters, run eager and compiled
packing isolation/parity canaries, and backpropagate the longest actual training
batch without changing adapters. Packing canaries retain the strict 5% gradient
gate and fixed leakage tolerances; selected finite acceptance never overrides
these isolation gates. Existing FlashQLA-versus-FLA and general compiler/partition
canaries preserve strict results and the already-selected `selected_finite`
policy. Every logical update checks missing/nonfinite master gradients.
Unsupported objectives, multiple devices, incompatible kernels, leakage,
nonfinite values or OOM fail before an optimizer update; never shrink the recipe
or substitute another backend to conceal a failure.

Evidence: the matched ten-step packed BF16 screen averaged 5.1452 seconds/update
without checkpointing at 147.14 GiB peak allocated memory, versus 5.7163 seconds
with checkpointing. Increasing the checkpointed token budget to 32,768 averaged
5.5162 seconds at 111.04 GiB. All packing isolation gates passed and all measured
training values were finite. See the [packing finding](../findings/bf16_sequence_packing.md).
These are bounded systems results; longer convergence/held-out quality remain
unmeasured. Planning estimate for the current 21,837-row one-epoch corpus is
683 updates, about 59 minutes of steady compute; allow additional setup/saving
and full-population length variation.

Historical original-FLA profiles remain unchanged. The preceding NF4/FlashQLA
default is available as `qwen35_4b_b200_flashqla`. Frozen JSON jobs and historical
paired packing benchmarks retain their original recipes. H100 defaults retain
their documented QLoRA/FLA recipe.

Ordinary Trainer integration is validated with focused CPU tests for scoped
settings, per-example gradients, arbitrary packed sizes and partial final
batches. The two-step B200 smoke in
`experiments/monitoring_sequence_packing/default_recipe_smoke.yaml` checks the
normal training path against the same frozen initial adapter and 320-row cohort,
including native eager/compiled isolation gates, longest-row memory preflight,
finite gradients, zero padding and a changed FP32 master. It completed both
ordinary Trainer updates, logical batches `[32,32]`, 16 physical calls (including
14-example packs), no checkpointed layers and 145.95 GiB peak allocated memory.
Its longest actual input was 28,733 tokens. Eager/compiled packing gradient
relative L2 was 0.00888195/0.00697903, with zero measured leakage. The broader
eight-example singleton/packed gradient comparison differed by 21.1489%, while
mean losses were 1.064834/1.074136. That strict 5% result remains false and is
accepted only by the existing `selected_finite` partition policy; the stricter
packing isolation gates remain independent and mandatory. Do not infer general
gradient agreement from the smaller packing probe or two successful updates.
The smoke produced 24 Dynamo graphs without cache-exhaustion fallback. It
validates the ordinary training wiring, not warmed performance or convergence.
Validation: 196 focused CPU tests passed (eight unsupported diagnostics skipped);
48 affected CPU checks passed again after the metadata update. Ruff and staged
diff checks passed. Reports and the FP32 adapter are collected locally under
`results/bf16_packed_default_smoke/`.

The complete ordinary-training metadata contract passed validation, including
14-example packs exceeding the legacy padded batch-size cap. The saved adapter
contains 256 FP32 LoRA tensors; its collected checksum matches the B200 original.
Reports, configuration, logs and weights are locally checksummed.
